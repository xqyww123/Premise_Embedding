/*  Title:      Semantic_Embedding/src/scala/pide_state.scala
    Author:     Qiyuan Xu

Scala functions for accessing PIDE document state from ML:
  - Save theory file contents (read-only: returns file/source pairs for ML to write)
  - Command ID to file+line position mapping
  - Go-to-definition: resolve entity reference at a position to its definition position
  - Hover message: extract tooltip information at a position
*/

package isabelle.semantic_embedding

import isabelle._


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

  private def find_node_name(version: Document.Version, file_path: String)
      : Option[Document.Node.Name] =
    version.nodes.iterator.map(_._1).find(name => name.node == file_path)

  override def invoke(session: Session, args: List[Bytes]): List[Bytes] = {
    val (file_path, offset) =
      XML.Decode.pair(XML.Decode.string, XML.Decode.int)(
        YXML.parse_body(args.head.text))

    val state = session.get_state()
    val version = state.recent_finished.version.get_finished

    val result: (String, Int, Int, Int) = find_node_name(version, file_path) match {
      case Some(node_name) =>
        val snapshot = state.snapshot(node_name = node_name)
        val range = Text.Range(offset - 1, offset)

        val links = snapshot.cumulate[List[(String, Int, Int, Int)]](
          range, Nil, Markup.Elements(Markup.ENTITY), _ => {
            case (acc, Text.Info(_, XML.Elem(Markup(Markup.ENTITY, props), _))) =>
              props match {
                case Position.Item_Def_File(name, line, def_range) =>
                  Some((name, line, def_range.start, def_range.stop) :: acc)
                case Position.Item_Def_Id(id, def_range) =>
                  snapshot.find_command_position(id, def_range.start) match {
                    case Some(node_pos) =>
                      Some((node_pos.name, node_pos.line1, 0, 0) :: acc)
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
      case None => ("", 0, 0, 0)
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

    val node_name_opt =
      version.nodes.iterator.map(_._1).find(name => name.node == file_path)

    val result: String = node_name_opt match {
      case Some(node_name) =>
        val snapshot = state.snapshot(node_name = node_name)
        val range = Text.Range(offset - 1, offset)

        val tooltip_elements = Markup.Elements(
          Markup.ENTITY, Markup.TYPING, Markup.SORTING, Markup.ML_TYPING)

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
      case None => ""
    }

    List(Bytes(YXML.string_of_body(XML.Encode.string(result))))
  }
}


class PIDE_State_Functions extends Scala.Functions(
  Save_Thy_Files, Resolve_Positions, Goto_Definition, Hover_Message)
