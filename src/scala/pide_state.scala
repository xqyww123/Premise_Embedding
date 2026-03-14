/*  Title:      Semantic_Embedding/src/scala/pide_state.scala
    Author:     Qiyuan Xu

Scala functions for accessing PIDE document state from ML:
  - Save theory file contents (read-only: returns file/source pairs for ML to write)
  - Command ID to file+line position mapping
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

    val results: List[(String, Int)] =
      input.map { case (id, offset) =>
        snapshot.find_command_position(id, offset) match {
          case Some(node_pos) => (node_pos.name + "~", node_pos.line1)
          case None => ("", 0)
        }
      }

    val body =
      XML.Encode.list(
        XML.Encode.pair(XML.Encode.string, XML.Encode.int)
      )(results)
    List(Bytes(YXML.string_of_body(body)))
  }
}


class PIDE_State_Functions extends Scala.Functions(Save_Thy_Files, Resolve_Positions)
