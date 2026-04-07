/*  Title:      Semantic_Embedding/src/scala/pide_state.scala
    Author:     Qiyuan Xu

Scala functions for accessing PIDE document state from ML:
  - Save theory file contents (read-only: returns file/source pairs for ML to write)
  - Command ID to file+line position mapping
  - Go-to-definition: resolve entity reference at a position to its definition position
    (live PIDE first, then DB fallback via Build.read_theory)
  - Hover message: extract tooltip information at a position
    (live PIDE first, then DB fallback via Build.read_theory)
  - Command at position: retrieve source code of the command at a given position
*/

package isabelle.semantic_embedding

import isabelle._


/* Shared query helpers operating on any Document.Snapshot */

object PIDE_Query {
  private def symbol_offset_to_range(
    snapshot: Document.Snapshot, offset: Int
  ): Text.Range = {
    snapshot.snippet_command match {
      case Some(command) =>
        // Snippet: convert from symbol offset to decoded text offset
        val start = command.chunk.decode(offset)
        val stop = command.chunk.decode(offset + 1)
        Text.Range(start, stop)
      case None =>
        // Live PIDE: offsets are already in the right coordinate system
        Text.Range(offset - 1, offset)
    }
  }

  def goto_definition(snapshot: Document.Snapshot, offset: Int)
      : (String, Int, Int, Int) = {
    val range = symbol_offset_to_range(snapshot, offset)

    val links = snapshot.cumulate[List[(String, Int, Int, Int)]](
      range, Nil, Markup.Elements(Markup.ENTITY), _ => {
        case (acc, Text.Info(_, XML.Elem(Markup(Markup.ENTITY, props), _))) =>
          props match {
            case Position.Item_Def_File(name, line, def_range) =>
              Some((name, line, def_range.start, def_range.stop) :: acc)
            case Position.Item_Def_Id(id, def_range) =>
              snapshot.find_command(id) match {
                case Some((node, command)) =>
                  val name = command.node_name.node
                  val preceding_symbols =
                    node.commands.iterator.takeWhile(_ != command)
                      .map(_.chunk.range.stop).sum
                  val within_lines =
                    if (def_range.start <= 0) 0
                    else {
                      val decoded = command.chunk.decode(def_range.start)
                      Text.Range(0, decoded).try_substring(command.source) match {
                        case Some(text) => Library.count_newlines(text)
                        case None => 0
                      }
                    }
                  val line = node.command_start_line(command).getOrElse(1) + within_lines
                  Some((name, line,
                        preceding_symbols + def_range.start,
                        preceding_symbols + def_range.stop) :: acc)
                case None => None
              }
            case _ => None
          }
        case _ => None
      })

    links match {
      case Text.Info(_, result :: _) :: _ => result
      case _ => ("", 0, 0, 0)
    }
  }

  private val tooltip_elements = Markup.Elements(
    Markup.ENTITY, Markup.TYPING, Markup.SORTING, Markup.ML_TYPING)

  def entity_at_position(snapshot: Document.Snapshot, offset: Int)
      : (String, String) = {
    val range = symbol_offset_to_range(snapshot, offset)

    val results = snapshot.cumulate[(String, String)](
      range, ("", ""), Markup.Elements(Markup.ENTITY), _ => {
        case (("", ""), Text.Info(_, XML.Elem(Markup.Entity(kind, name), _)))
          if kind != "" && kind != Markup.ML_DEF =>
          Some((kind, name))
        case _ => None
      })

    results match {
      case Text.Info(_, result) :: _ if result != ("", "") => result
      case _ => ("", "")
    }
  }

  def command_at_position(node: Document.Node, offset: Int): (String, Int, Int) = {
    var pos = 1
    var result: (String, Int, Int) = ("", 0, 0)
    val it = node.commands.iterator
    while (it.hasNext && result._1.isEmpty) {
      val command = it.next()
      val len = command.chunk.range.stop
      if (offset >= pos && offset < pos + len)
        result = (command.source, pos, pos + len)
      pos += len
    }
    result
  }

  def command_at_position(snapshot: Document.Snapshot, offset: Int): (String, Int, Int) = {
    val range = symbol_offset_to_range(snapshot, offset)

    val spans = snapshot.cumulate[Option[Text.Range]](
      range, None, Markup.Elements(Markup.COMMAND_SPAN), _ => {
        case (None, Text.Info(info_range, _)) => Some(Some(info_range))
        case _ => None
      })

    spans match {
      case Text.Info(_, Some(span_range)) :: _ =>
        snapshot.snippet_command match {
          case Some(command) =>
            // DB snippet: ranges are in decoded text offsets, need to extract source substring
            val src = span_range.try_substring(command.source).getOrElse("")
            // Convert decoded text offsets back to symbol offsets for the return value
            val chunk = command.chunk
            // Invert: find symbol offsets that decode to span_range.start/stop
            // Use a linear scan of symbol positions
            val sym_start = symbol_offset_of_decoded(chunk, span_range.start)
            val sym_stop = symbol_offset_of_decoded(chunk, span_range.stop)
            (src, sym_start, sym_stop)
          case None =>
            // Live PIDE: shouldn't reach here, but handle gracefully
            ("", 0, 0)
        }
      case _ => ("", 0, 0)
    }
  }

  /** Find the 1-based symbol offset whose decoded text offset is >= target.
    * This inverts Text_Chunk.decode for the purpose of returning symbol offsets. */
  private def symbol_offset_of_decoded(chunk: Symbol.Text_Chunk, target: Int): Int = {
    // Binary search: find smallest symbol offset s such that chunk.decode(s) >= target
    var lo = 1
    var hi = chunk.range.stop + 1  // upper bound: one past the last symbol
    while (lo < hi) {
      val mid = (lo + hi) / 2
      if (chunk.decode(mid) < target) lo = mid + 1
      else hi = mid
    }
    lo
  }

  def hover_message(snapshot: Document.Snapshot, offset: Int): String = {
    val range = symbol_offset_to_range(snapshot, offset)

    val results = snapshot.cumulate[List[String]](
      range, Nil, tooltip_elements, _ => {
        case (tips, Text.Info(_, XML.Elem(Markup.Entity(kind, name), _)))
          if kind != "" && kind != Markup.ML_DEF =>
          val kind1 = Word.implode(Word.explode('_', kind))
          val txt =
            if (name == "") kind1
            else if (kind1 == "") name
            else kind1 + " " + quote(name)
          Some(txt :: tips)

        case (tips, Text.Info(_, XML.Elem(Markup(name, _), body)))
          if name == Markup.TYPING || name == Markup.SORTING =>
          val body_text = XML.content(Pretty.formatted(body))
          Some((":: " + body_text) :: tips)

        case (tips, Text.Info(_, XML.Elem(Markup(Markup.ML_TYPING, _), body))) =>
          val body_text = XML.content(Pretty.formatted(body))
          Some(("ML: " + body_text) :: tips)

        case _ => None
      })

    results match {
      case Text.Info(_, tips) :: _ => tips.reverse.mkString("\n")
      case _ => ""
    }
  }
}


/* DB snapshot loading and caching */

object DB_Snapshots {
  // digest -> (session_name, theory_name)
  private var digest_index: Map[String, (String, String)] = Map.empty
  private var indexed_sessions: Set[String] = Set.empty

  // theory_name -> snapshot
  private var snapshot_cache: Map[String, Document.Snapshot] = Map.empty

  // Cached Store instance (shared Term.Cache for string interning across snapshots)
  // One Session per Isabelle process, so this is effectively a singleton cache.
  private var cached_store: Option[(Session, Store)] = None

  private def get_store(session: Session): Store = synchronized {
    cached_store match {
      case Some((s, store)) if s eq session => store
      case _ =>
        val store = Store(session.session_options)
        cached_store = Some((session, store))
        store
    }
  }

  // file_path -> sha1 digest (files are immutable once built)
  private var file_digest_cache: Map[String, String] = Map.empty

  private def file_digest(path: Path): String = synchronized {
    val key = path.implode
    file_digest_cache.get(key) match {
      case Some(d) => d
      case None =>
        val d = SHA1.digest(path).toString
        file_digest_cache += (key -> d)
        d
    }
  }

  private def index_session(store: Store, session_name: String): Unit = synchronized {
    if (indexed_sessions.contains(session_name)) return
    indexed_sessions += session_name

    store.try_open_database(session_name, server_mode = false) match {
      case Some(db) =>
        try {
          // Query only name and digest columns for .thy files — skip reading file bodies
          val thy_digests = db.execute_query_statement(
            Store.private_data.Sources.table.select(
              List(Store.private_data.Sources.name, Store.private_data.Sources.digest),
              sql = SQL.where_and(
                Store.private_data.Sources.session_name.equal(session_name),
                Store.private_data.Sources.name.ident + " LIKE '%.thy'")),
            List.from[(String, String)],
            res => (res.string(Store.private_data.Sources.name),
                    res.string(Store.private_data.Sources.digest)))
          for ((name, digest) <- thy_digests) {
            val base = Library.try_unsuffix(".thy", Path.explode(name).file_name).getOrElse("")
            if (base.nonEmpty) {
              val theory_name = session_name + "." + base
              digest_index += (digest -> (session_name, theory_name))
            }
          }
        }
        finally { db.close() }
      case None =>
        Output.warning("DB_Snapshots.index_session: " + session_name + " — no database found")
    }
  }

  private def load_snapshot(
    store: Store,
    session_name: String,
    theory_name: String
  ): Option[Document.Snapshot] = synchronized {
    snapshot_cache.get(theory_name) match {
      case some @ Some(_) => some
      case None =>
        try {
          val result =
            using(Export.open_session_context0(store, session_name)) { session_context =>
              val theory_context = session_context.theory(theory_name)
              Build.read_theory(theory_context)
            }
          result.foreach(snapshot => snapshot_cache += (theory_name -> snapshot))
          result
        }
        catch {
          case exn: Exception =>
            Output.warning("DB_Snapshots.load_snapshot: exception — " + exn.getMessage)
            None
        }
    }
  }

  def get_snapshot(session: Session, file_path: String): Option[Document.Snapshot] = {
    val store = get_store(session)
    val sessions_structure = session.resources.sessions_structure
    val current_session = session.resources.session_base.session_name
    val ancestors = sessions_structure.build_hierarchy(current_session)

    // Index all ancestor sessions
    for (name <- ancestors) index_session(store, name)

    // Compute file digest and look up
    val file = Path.explode(file_path)
    if (!file.is_file) {
      Output.warning("DB_Snapshots.get_snapshot: file does not exist: " + file.implode)
      return None
    }
    val digest = file_digest(file)

    digest_index.get(digest) match {
      case Some((session_name, theory_name)) =>
        load_snapshot(store, session_name, theory_name)
      case None => None
    }
  }
}


object Save_Thy_Files extends Scala.Fun("pide_state.save_thy_files", thread = true)
  with Scala.Single_Fun
{
  val here = Scala_Project.here

  private var last_max_id: Map[String, Long] = Map.empty

  override def invoke(session: Session, args: List[Bytes]): List[Bytes] = {
    val version = session.get_state().recent_finished.version.get_finished

    val written: List[String] =
      (for {
        (name, node) <- version.nodes.iterator
        if name.is_theory
        src = node.source
        if src.nonEmpty
      } yield {
        val max_id = node.commands.iterator.map(_.id).maxOption.getOrElse(0L)
        val cached = last_max_id.getOrElse(name.node, -1L)
        if (max_id == cached) None
        else {
          val backup = Path.explode(name.node + "~")
          val existing = try { File.read(backup) } catch { case _: Exception => "" }
          if (existing == src) {
            last_max_id += (name.node -> max_id)
            None
          }
          else {
            File.write(backup, src)
            last_max_id += (name.node -> max_id)
            Some(backup.implode)
          }
        }
      }).flatten.toList

    val body = XML.Encode.list(XML.Encode.string)(written)
    List(Bytes(YXML.string_of_body(body)))
  }
}


object Resolve_Positions extends Scala.Fun("pide_state.resolve_positions", thread = true)
  with Scala.Single_Fun
{
  val here = Scala_Project.here

  override def invoke(session: Session, args: List[Bytes]): List[Bytes] = {
    val input =
      XML.Decode.list(XML.Decode.pair(XML.Decode.long, XML.Decode.int))(
        YXML.parse_body(args.head.text))

    val snapshot = session.get_state().snapshot()

    // Init: resolve commands, collect unique nodes
    val resolved = new scala.collection.mutable.HashMap[Long, (Document.Node, Command)]
    val nodes_to_scan = new scala.collection.mutable.LinkedHashSet[Document.Node]
    for ((id, _) <- input) {
      if (!resolved.contains(id)) {
        snapshot.find_command(id) match {
          case Some((node, command)) =>
            resolved(id) = (node, command)
            nodes_to_scan += node
          case None =>
        }
      }
    }

    // Scan: single pass per node to compute preceding symbol counts
    val needed_ids = resolved.values.map(_._2.id).toSet
    val preceding_symbols_map = new scala.collection.mutable.HashMap[Document_ID.Command, Int]
    for (node <- nodes_to_scan) {
      var symbols = 0
      for (command <- node.commands.iterator) {
        if (needed_ids.contains(command.id)) {
          preceding_symbols_map(command.id) = symbols
        }
        symbols += command.chunk.range.stop
      }
    }

    // Resolve: look up precomputed data, compute within-command line offset
    val results: List[(String, (Int, Int))] =
      input.map { case (id, offset) =>
        resolved.get(id) match {
          case Some((node, command)) =>
            val preceding_symbols = preceding_symbols_map.getOrElse(command.id, 0)
            val start_line = node.command_start_line(command).getOrElse(1)
            val within_lines =
              if (offset <= 1) 0
              else {
                val decoded = command.chunk.decode(offset)
                val range = Text.Range(0, decoded)
                range.try_substring(command.source) match {
                  case Some(text) => Library.count_newlines(text)
                  case None => 0
                }
              }
            (command.node_name.node, (start_line + within_lines, preceding_symbols))
          case None => ("", (0, 0))
        }
      }

    val body =
      XML.Encode.list(
        XML.Encode.pair(XML.Encode.string,
          XML.Encode.pair(XML.Encode.int, XML.Encode.int))
      )(results)
    List(Bytes(YXML.string_of_body(body)))
  }
}


object Goto_Definition extends Scala.Fun("pide_state.goto_definition", thread = true)
  with Scala.Single_Fun
{
  val here = Scala_Project.here

  override def invoke(session: Session, args: List[Bytes]): List[Bytes] = {
    val (file_path, offset) =
      XML.Decode.pair(XML.Decode.string, XML.Decode.int)(
        YXML.parse_body(args.head.text))

    val state = session.get_state()
    val version = state.recent_finished.version.get_finished

    // Try live PIDE state first
    val live_result: (String, Int, Int, Int) =
      version.nodes.iterator.map(_._1).find(_.node == file_path) match {
        case Some(node_name) =>
          PIDE_Query.goto_definition(state.snapshot(node_name = node_name), offset)
        case None => ("", 0, 0, 0)
      }

    // Fall back to DB if live returns nothing
    val result =
      if (live_result != ("", 0, 0, 0)) live_result
      else {
        DB_Snapshots.get_snapshot(session, file_path) match {
          case Some(snapshot) =>
            PIDE_Query.goto_definition(snapshot, offset)
          case None =>
            Output.warning("Goto_Definition: no DB snapshot found")
            ("", 0, 0, 0)
        }
      }

    val body = {
      val (a, b, c, d) = result
      XML.Encode.pair(XML.Encode.string,
        XML.Encode.pair(XML.Encode.int,
          XML.Encode.pair(XML.Encode.int, XML.Encode.int)))((a, (b, (c, d))))
    }
    List(Bytes(YXML.string_of_body(body)))
  }
}


object Hover_Message extends Scala.Fun("pide_state.hover_message", thread = true)
  with Scala.Single_Fun
{
  val here = Scala_Project.here

  override def invoke(session: Session, args: List[Bytes]): List[Bytes] = {
    val (file_path, offset) =
      XML.Decode.pair(XML.Decode.string, XML.Decode.int)(
        YXML.parse_body(args.head.text))

    val state = session.get_state()
    val version = state.recent_finished.version.get_finished

    // Try live PIDE state first
    val live_result: String =
      version.nodes.iterator.map(_._1).find(_.node == file_path) match {
        case Some(node_name) =>
          PIDE_Query.hover_message(state.snapshot(node_name = node_name), offset)
        case None => ""
      }

    // Fall back to DB if live returns nothing
    val result =
      if (live_result.nonEmpty) live_result
      else {
        DB_Snapshots.get_snapshot(session, file_path) match {
          case Some(snapshot) => PIDE_Query.hover_message(snapshot, offset)
          case None => ""
        }
      }

    List(Bytes(YXML.string_of_body(XML.Encode.string(result))))
  }
}


object Entity_At_Position extends Scala.Fun("pide_state.entity_at_position", thread = true)
  with Scala.Single_Fun
{
  val here = Scala_Project.here

  override def invoke(session: Session, args: List[Bytes]): List[Bytes] = {
    val (file_path, offset) =
      XML.Decode.pair(XML.Decode.string, XML.Decode.int)(
        YXML.parse_body(args.head.text))

    val state = session.get_state()
    val version = state.recent_finished.version.get_finished

    val live_result: (String, String) =
      version.nodes.iterator.map(_._1).find(_.node == file_path) match {
        case Some(node_name) =>
          PIDE_Query.entity_at_position(state.snapshot(node_name = node_name), offset)
        case None => ("", "")
      }

    val result =
      if (live_result != ("", "")) live_result
      else {
        DB_Snapshots.get_snapshot(session, file_path) match {
          case Some(snapshot) => PIDE_Query.entity_at_position(snapshot, offset)
          case None => ("", "")
        }
      }

    val body = XML.Encode.pair(XML.Encode.string, XML.Encode.string)(result)
    List(Bytes(YXML.string_of_body(body)))
  }
}


object Command_At_Position extends Scala.Fun("pide_state.command_at_position", thread = true)
  with Scala.Single_Fun
{
  val here = Scala_Project.here

  override def invoke(session: Session, args: List[Bytes]): List[Bytes] = {
    val (file_path, offset) =
      XML.Decode.pair(XML.Decode.string, XML.Decode.int)(
        YXML.parse_body(args.head.text))

    val state = session.get_state()
    val version = state.recent_finished.version.get_finished

    // Try live PIDE state first
    val live_result: (String, Int, Int) =
      version.nodes.iterator.map(_._1).find(_.node == file_path) match {
        case Some(node_name) =>
          PIDE_Query.command_at_position(version.nodes(node_name), offset)
        case None => ("", 0, 0)
      }

    // Fall back to DB if live returns nothing
    val result =
      if (live_result != ("", 0, 0)) live_result
      else {
        DB_Snapshots.get_snapshot(session, file_path) match {
          case Some(snapshot) =>
            PIDE_Query.command_at_position(snapshot, offset)
          case None => ("", 0, 0)
        }
      }

    val body = XML.Encode.pair(XML.Encode.string,
      XML.Encode.pair(XML.Encode.int, XML.Encode.int))(
        (result._1, (result._2, result._3)))
    List(Bytes(YXML.string_of_body(body)))
  }
}


object Get_Session_Databases extends Scala.Fun("pide_state.get_session_databases", thread = true)
  with Scala.Single_Fun
{
  val here = Scala_Project.here

  override def invoke(session: Session, args: List[Bytes]): List[Bytes] = {
    val store = Store(session.session_options)
    val sessions_structure = session.resources.sessions_structure
    val current_session = session.resources.session_base.session_name
    val ancestors = sessions_structure.build_hierarchy(current_session)

    val results: List[(String, String)] = ancestors.flatMap { name =>
      val s = store.get_session(name)
      s.log_db.map(path => (name, File.standard_path(path)))
    }

    val body = XML.Encode.list(
      XML.Encode.pair(XML.Encode.string, XML.Encode.string))(results)
    List(Bytes(YXML.string_of_body(body)))
  }
}


/* Feasibility probe: dump COMMAND_SPAN keyword + ENTITY def/ref markup
   for the command at a given position.  Returns a human-readable string. */

object Probe_Command_Header extends Scala.Fun("pide_state.probe_command_header", thread = true)
  with Scala.Single_Fun
{
  val here = Scala_Project.here

  override def invoke(session: Session, args: List[Bytes]): List[Bytes] = {
    val (file_path, offset) =
      XML.Decode.pair(XML.Decode.string, XML.Decode.int)(
        YXML.parse_body(args.head.text))

    val state = session.get_state()
    val version = state.recent_finished.version.get_finished
    val result = new StringBuilder

    version.nodes.iterator.map(_._1).find(_.node == file_path) match {
      case Some(node_name) =>
        val node = version.nodes(node_name)
        val snapshot = state.snapshot(node_name = node_name)

        // Find command via node iteration
        var pos = 1
        var found: Option[(Command, Int)] = None
        val it = node.commands.iterator
        while (it.hasNext && found.isEmpty) {
          val command = it.next()
          val len = command.chunk.range.stop
          if (offset >= pos && offset < pos + len)
            found = Some((command, pos))
          pos += len
        }

        found match {
          case Some((command, cmd_pos)) =>
            val len = command.chunk.range.stop
            result.append("keyword (span.name): " + command.span.name + "\n")
            result.append("source: " + command.source.take(100).replace("\n", "\\n") + "\n")
            result.append("span: [" + cmd_pos + ", " + (cmd_pos + len) + ")\n")

            // Search for all ENTITY markup in the command range
            val cmd_range = Text.Range(cmd_pos - 1, cmd_pos - 1 + len)
            val entities = snapshot.cumulate[List[(String, String, String)]](
              cmd_range, Nil, Markup.Elements(Markup.ENTITY), _ => {
                case (acc, Text.Info(_, XML.Elem(markup @ Markup(Markup.ENTITY, props), _))) =>
                  val kind = Markup.Kind.get(props)
                  val name = Markup.Name.get(props)
                  val tag =
                    if (Markup.Entity.Def.unapply(markup).isDefined) "DEF"
                    else if (Markup.Entity.Ref.unapply(markup).isDefined) "REF"
                    else "???"
                  Some((kind, name, tag) :: acc)
                case _ => None
              })

            entities match {
              case Text.Info(_, ents) :: _ =>
                result.append("entities (" + ents.length + "):\n")
                for ((kind, name, tag) <- ents.reverse)
                  result.append("  [" + tag + "] " + kind + " " + name + "\n")
              case _ =>
                result.append("entities: none\n")
            }

          case None =>
            result.append("no command at offset " + offset + " in " + file_path)
        }

      case None =>
        result.append("no live node for: " + file_path)
    }

    List(Bytes(YXML.string_of_body(XML.Encode.string(result.toString))))
  }
}


class PIDE_State_Functions extends Scala.Functions(
  Save_Thy_Files, Resolve_Positions, Goto_Definition, Hover_Message,
  Entity_At_Position, Command_At_Position, Get_Session_Databases,
  Probe_Command_Header)
