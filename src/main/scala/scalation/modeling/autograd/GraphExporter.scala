
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Thu Nov 13 11:42:31 EST 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: Computational Graph Export Utilities
 *
 *  Provides utilities for exporting autograd computation graphs into multiple
 *  visualization formats (DOT/GraphViz, Mermaid, D3.js JSON).  Starting from a
 *  root `Variabl`, the graph builder performs a reachability scan, determines
 *  variable/function roles, detects shapes, assigns depths for layout, and
 *  produces a structured `GraphModel` representation.
 *
 *  Exporters:
 *    - `toDot`     : GraphViz DOT format with optional depth clustering, shapes,
 *                     gradients, and legend rendering.
 *    - `toMermaid` : Mermaid Flowchart syntax.
 *    - `toJson`    : D3.js-friendly node/edge JSON.
 *    - `writeDot`, `writeAll` : Convenience writers for filesystem output.
 */

package scalation
package modeling
package autograd

import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets

import scala.util.Try

import scalation.database.graph_pm.TopSort
import scalation.mathstat.TensorD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** GraphExporter generates a computation graph visualization from a root
 *  `Variabl`.  The graph includes variables, functions, dependency edges,
 *  tensor shapes, and optional gradient annotations.  The resulting graph can
 *  be serialized to DOT, Mermaid, or JSON formats for visualization.
 */
object GraphExporter:
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rendering options for DOT output.
     *  @param showAnnotations  whether to annotate nodes that have stored gradients
     *  @param edgeShapes       whether to label edges with tensor shapes
     *  @param nodeShapes       whether to display tensor shapes inside nodes
     *  @param colorScheme      color theme for rendering (reserved for future use)
     *  @param groupBy          grouping mode ("depth" or "none")
     *  @param showLegend       whether to include a legend cluster in the DOT output
     */
    case class RenderOptions (showAnnotations: Boolean = true, edgeShapes: Boolean = true,
                              nodeShapes: Boolean = true, colorScheme: String = "default",
                              groupBy: String = "depth", showLegend: Boolean = true)
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** A variable node in the exported computation graph.
     *  @param id        unique identifier for the variable
     *  @param isParam   whether this variable represents a trainable parameter
     *  @param isOutput  whether this variable is the graph’s final output
     *  @param shape     tensor shape of the variable
     *  @param name      optional user-defined name
     *  @param grad      optional stored gradient tensor
     */
    case class VarNode (id: String, isParam: Boolean,
                        isOutput: Boolean, shape: List[Int],
                        name: Option[String], grad: Option[TensorD])
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** A function node (operation) in the computation graph.
     *  @param id     unique identifier for the function
     *  @param op     operation name (e.g., "add", "matmul")
     *  @param attrs  operator-specific attributes
     *  @param shape  tensor shape of the function output
     *  @param depth  depth level for layered graph layout
     */
    case class FuncNode (id: String, op: String,
                         attrs: Map[String,String], shape: List[Int], depth: Int)
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** A directed graph edge.
     *  @param src   source node identifier
     *  @param dst   destination node identifier
     *  @param kind  edge type (currently only "data")
     */
    case class Edge (src: String, dst: String, kind: String) 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Internal model of the full computation graph.
     *  @param vars   all variable nodes
     *  @param funcs  all function nodes
     *  @param edges  all dependency edges
     *  @param root   id of the root output variable
     */
    case class GraphModel (vars: Seq [VarNode], funcs: Seq [FuncNode],
                           edges: Seq [Edge], root: String)
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a computation graph starting from a root `Variabl`.
     *  Traverses all dependent variables, topologically sorts functions,
     *  assigns depth levels, and constructs variable/function nodes and edges.
     *  @param root        the root output variable
     *  @param includeGrad whether to attach stored gradients to variable nodes
     *  @return a structured `GraphModel` for visualization
     */
    // FIX - Doesn't work with RNNFused yet (possibly due to name mismatch)
    def build (root: Variabl, includeGrad: Boolean = true): GraphModel =
        // Collect all variables reachable from the root
        val vars = scala.collection.mutable.LinkedHashSet [Variabl] ()
        def visit (v: Variabl): Unit =
            if ! vars.contains (v) then
                vars += v
                v.gradFn.foreach (f => f.inputs.foreach (visit))
        visit (root)
        
        // Topo-sorted function nodes (inputs → outputs)
        val fns = TopSort.topSortFunctions (root)
        
        // Map each Function -> its output Variabl
        val funcOut: Map[Function, Variabl] =
            vars.flatMap (v => v.gradFn.map (_ -> v)).toMap
        
        // Depth assignment (for nicer horizontal layering)
        val depthCache = scala.collection.mutable.HashMap [Any, Int] ()
        
        def depthOfVar(v: Variabl): Int = v.gradFn match
            case None    => 0
            case Some(f) => depthCache.getOrElseUpdate (f, depthOfFunc(f))
        
        def depthOfFunc (f: Function): Int =
            depthCache.getOrElseUpdate (f,
                (if f.inputs.isEmpty then 0 else f.inputs.map(depthOfVar).max) + 1)
        
        val funcNodes = fns.map { f =>
            val out = funcOut.getOrElse (f,
                throw new IllegalStateException(s"No output var for Function id=${f.id} (${f.opName})"))
            FuncNode (id = s"F${f.id}", op = f.opName,
                      attrs = f.attributes, shape = out.data.shape, depth = depthOfFunc(f))
        }
        
        val varNodes = vars.toSeq.map { v =>
            val isOut    = (v eq root)
            val isParam  = v.gradFn.isEmpty && v.name.exists(n =>
                val ln = n.toLowerCase
                ln.contains ("weight") || ln.contains ("bias") ||
                ln.startsWith ("w_")   || ln.startsWith ("b_"))
            VarNode (id = s"V${System.identityHashCode(v)}",
                     isParam = isParam, isOutput= isOut,
                     shape = v.data.shape, name = v.name,
                     grad = if includeGrad then Some(v.grad) else None)
        }
        
        // Stable mapping Var -> id (same order used to build varNodes)
        val idMap: Map [Variabl, String] =
            vars.toSeq.zip (varNodes.map (_.id)).toMap
        
        def idOf(v: Variabl): String = idMap (v)
        
        // Data edges: Var -> Func (inputs) and Func -> Var (output)
        val edgesData: Seq [Edge] = fns.flatMap { f =>
            val fId = s"F${f.id}"
            val out = funcOut(f)
            val ins = f.inputs.map(in => Edge(idOf(in), fId, "data"))
            ins :+ Edge (fId, idOf(out), "data") }
        
        GraphModel (varNodes, funcNodes, edgesData, idOf (root))
    end build
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Escape special characters in a string for safe usage in DOT/GraphViz syntax.
     *  Specifically, this method replaces backslashes (`\`) with double backslashes (`\\`)
     *  and double quotes (`"`) with escaped double quotes (`\"`).
     *  @param s  the input string to be escaped
     *  @return the escaped string with special characters replaced
     */
    private def esc (s: String): String =
        s.replace("\\", "\\\\").replace ("\"", "\\\"")
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert a graph model into GraphViz DOT format.
     *  Supports optional depth-based clustering, shape labels, gradient tags,
     *  and legend rendering.
     *  @param g     the computation graph
     *  @param opts  rendering options
     *  @return DOT string representation
     */
    def toDot (g: GraphModel, opts: RenderOptions = RenderOptions()): String =
        val sb = new StringBuilder
        sb ++= "digraph ComputationGraph {\n" +
               "  rankdir=LR;\n" +
               "  compound=true;\n" +
               "  node [fontname=\"Helvetica\"];\n"
        
/*
        // For edge shape labels and leaf detection
        val varMap: Map [String, VarNode] = g.vars.map (v => v.id -> v).toMap
        val producedVarIds: Set [String] =
            g.edges.collect {
                case Edge(src, dst, "data") if src.startsWith ("F") && dst.startsWith ("V") => dst
            }.toSet
*/
        
        // Functions (either grouped by depth or flat)
        if opts.groupBy == "depth" then
            g.funcs.groupBy (_.depth).toSeq.sortBy (_._1).foreach { case (d, layer) =>
                sb ++= s"  subgraph cluster_depth_$d {\n    label=\"Layer $d\"; style=dashed; color=gray70;\n"
                layer.foreach { f =>
                    val shapeLine = if opts.nodeShapes then s"shape=${f.shape.mkString("x")}" else ""
                    val attrLines =
                        (if shapeLine.nonEmpty then Seq(shapeLine) else Seq.empty) ++
                          f.attrs.map { case (k,v) => s"$k=$v" }
                    val extra = if attrLines.nonEmpty then attrLines.mkString("\\n") else ""
                    sb ++=
                      s"""    ${f.id} [shape=box, style=filled, fillcolor="#FFF7D6", label="${esc(f.op)}${if extra.nonEmpty then "\n"+extra else ""}"];\n"""
                }
                sb ++= "  }\n"
            }
        else
            g.funcs.sortBy (_.depth).foreach { f =>
                val shapeLine = if opts.nodeShapes then s"shape=${f.shape.mkString("x")}" else ""
                val attrLines =
                    (if shapeLine.nonEmpty then Seq (shapeLine) else Seq.empty) ++
                      f.attrs.map{ case (k,v) => s"$k=$v" }
                val extra = if attrLines.nonEmpty then attrLines.mkString("\\n") else ""
                sb ++=
                  s"""  ${f.id} [shape=box, style=filled, fillcolor="#FFF7D6", label="${esc(f.op)}${if extra.nonEmpty then "\n"+extra else ""}"];\n"""
            }
        
        // Variables (color by role)
        g.vars.foreach { v =>
            val isLeaf = !v.isOutput && !v.isParam && !g.edges.exists(e => e.src.startsWith("F") && e.dst == v.id)
            val fill =
                if v.isOutput then "#FFCCE0"
                else if v.isParam then "#FFE9AA"
                else if isLeaf then "#D5F5D5"
                else "#E0F1FF"
            val style   = if v.isParam then "doublecircle" else "ellipse"
            val gradTag = if opts.showAnnotations && v.grad.isDefined then "\\n∂L stored" else ""
            val nm      = v.name.map (esc).getOrElse ("")
            val shapeLn = if opts.nodeShapes then s"\n${v.shape.mkString("x")}" else ""
            sb ++= s"  ${v.id} [shape=$style, style=filled, fillcolor=\"$fill\", label=\"${nm}$shapeLn$gradTag\"];\n"
        }
        
        // Edges (optionally label with tensor shapes)
        g.edges.foreach { e =>
            if e.kind == "data" then
                val labelOpt =
                    if opts.edgeShapes then
                        if e.src.startsWith ("V") && e.dst.startsWith ("F") then g.vars.find (_.id == e.src).map(_.shape.mkString("x"))
                        else if e.src.startsWith ("F") && e.dst.startsWith ("V") then g.vars.find (_.id == e.dst).map(_.shape.mkString("x"))
                        else None
                    else None
                val labelStr = labelOpt.map(l => s" label=\"${esc(l)}\"").getOrElse("")
                sb ++= s"  ${e.src} -> ${e.dst} [color=black$labelStr];\n"
        }
        if opts.showLegend then
            sb ++= "  subgraph cluster_legend {\n"
            sb ++= "    label=\"\"; style=rounded; color=gray60; fontsize=10;\n"
            // Compact HTML label (shape=plain) single block (no leading/trailing blank lines inside <>)
            sb ++= "    legend_info [shape=plain, label=<<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLPADDING=\"2\">" +
                "<TR><TD><B>Legend</B></TD></TR>" +
                "<TR><TD ALIGN=\"LEFT\">Function: operation (box, pale yellow)</TD></TR>" +
                "<TR><TD ALIGN=\"LEFT\">Param: parameter variable (gold, doublecircle)</TD></TR>" +
                "<TR><TD ALIGN=\"LEFT\">Leaf: leaf input variable (green)</TD></TR>" +
                "<TR><TD ALIGN=\"LEFT\">Intermediate: intermediate variable (light blue)</TD></TR>" +
                "<TR><TD ALIGN=\"LEFT\">Output: final output variable (pink)</TD></TR>" +
                "<TR><TD ALIGN=\"LEFT\">Edge label: tensor shape</TD></TR>" +
                "<TR><TD ALIGN=\"LEFT\">Node suffix: ∂L stored (gradient available)</TD></TR>" +
                "</TABLE>>];\n"
            sb ++= "  }\n"
        sb ++= "}\n"
        sb.toString()
    end toDot
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward-compatible DOT exporter using only a gradient toggle.
     *  @param g         the computation graph
     *  @param showGrad  whether to annotate gradient availability
     */
    def toDot (g: GraphModel, showGrad: Boolean): String =
        toDot (g, RenderOptions(showAnnotations = showGrad))
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Export the computation graph in Mermaid Flowchart syntax.
     *  Variables become rounded nodes; functions become double-curly nodes.
     *  @param g  the graph model
     *  @return Mermaid flowchart string
     */
    def toMermaid (g: GraphModel): String =
        val sb = new StringBuilder ("flowchart LR\n")
        g.vars.foreach { v =>
            val nm = v.name.getOrElse (v.id)
            sb ++= s"""  ${v.id}(["$nm\\n${v.shape.mkString("x")}"])\n"""
        }
        g.funcs.foreach { f =>
            sb ++= s"""  ${f.id}{{"${f.op}\\n${f.shape.mkString("x")}"]}}\n"""
        }
        g.edges.filter (_.kind == "data").foreach { e =>
            sb ++= s"  ${e.src} --> ${e.dst}\n"
        }
        sb.toString
    end toMermaid
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Export the graph in JSON format for D3.js visualizations.
     *  @param g  the graph model
     *  @return JSON string containing nodes and edges
     */
    def toJson (g: GraphModel): String =
        val nodes =
            (g.vars.map { v =>
                s"""{"id":"${v.id}","type":"var","name":${v.name.fold("null")(n => s"\"${esc(n)}\"")},"shape":"${v.shape.mkString("x")}","isParam":${v.isParam},"isOutput":${v.isOutput}}"""
            } ++ g.funcs.map { f =>
                val attrs = f.attrs.map{ case (k,v) => s""""${esc(k)}":"${esc(v)}""" }.mkString(",")
                s"""{"id":"${f.id}","type":"func","op":"${esc(f.op)}","shape":"${f.shape.mkString("x")}","depth":${f.depth},"attrs":{${attrs}}}"""
            }).mkString (",")

        val edges =
            g.edges.filter (_.kind == "data")
              .map(e => s"""{"source":"${e.src}","target":"${e.dst}"}""")
              .mkString (",")
        s"""{"nodes":[${nodes}],"edges":[${edges}]}"""
    end toJson
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Write DOT output to a file, with optional GraphViz SVG rendering.
     *  @param root       root variable of the computation graph
     *  @param path       target .dot file path
     *  @param opts       rendering options
     *  @param renderSvg  whether to also generate an SVG via `dot -Tsvg`
     *  @return path of written file or SVG file
     */
    def writeDot (root: Variabl, path: String,
                  opts: RenderOptions = RenderOptions (),
                  renderSvg: Boolean = false): Try [String] = Try {
        val p = Paths.get (path)
        val parent = p.getParent
        if parent != null && ! Files.exists (parent) then Files.createDirectories (parent)
        val g   = build (root)
        val dot = toDot (g, opts)
        Files.write (p, dot.getBytes (StandardCharsets.UTF_8))
        if renderSvg then
            val out = path.stripSuffix (".dot") + ".svg"
            val pb  = new ProcessBuilder ("dot", "-Tsvg", path, "-o", out)
            pb.inheritIO ().start ().waitFor ()
            out
        else path
    }
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Write all supported graph formats (DOT, SVG, Mermaid, JSON) into a directory.
     *  @param root      root variable
     *  @param dir       output directory
     *  @param baseName  base filename without extension
     *  @param opts      rendering options
     *  @param svg       whether to export SVG
     *  @param mermaid   whether to export Mermaid
     *  @param json      whether to export JSON
     *  @return sequence of file paths written
     */
    def writeAll (root: Variabl, dir: String,
                  baseName: String, opts: RenderOptions = RenderOptions (),
                  svg: Boolean = true, mermaid: Boolean = true,
                  json: Boolean = true): Try [Seq [String]] = Try {
        val outDir = Paths.get (dir)
        if !Files.exists (outDir) then Files.createDirectories (outDir)
        val results = scala.collection.mutable.ArrayBuffer [String] ()

        val dotPath = outDir.resolve (baseName + ".dot").toString
        writeDot (root, dotPath, opts, renderSvg = svg).foreach( results += _)

        val g = build (root)
        if mermaid then
            val mer = toMermaid (g)
            val mp  = outDir.resolve (baseName + ".mmd")
            Files.write(mp, mer.getBytes(StandardCharsets.UTF_8))
            results += mp.toString

        if json then
            val js = toJson (g)
            val jp = outDir.resolve (baseName + ".json")
            Files.write(jp, js.getBytes (StandardCharsets.UTF_8))
            results += jp.toString

        results.toSeq
    }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward-compatible alias for `writeDot`.
     *  @param root   root variable
     *  @param path   target DOT file
     *  @param render whether to also produce SVG
     */
    def makeDot(root: Variabl, path: String, render: Boolean = false): Try[String] =
        writeDot(root, path, RenderOptions(), renderSvg = render)

end GraphExporter

