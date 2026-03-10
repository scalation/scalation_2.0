
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo, John Miller
 *  @version 2.0
 *  @date    Sat Sep 20 17:13:38 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Support for Converting ScalaTion Equations and Matrices to LaTeX
 */

package scalation
package mathstat

import scala.collection.mutable.{LinkedHashSet => LSET}
import scala.util.matching.Regex

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Scala2LaTeX object provides methods for converting ScalaTion Equations
 *  into LaTeX.
 */
object Scala2LaTeX:

    /** Customize LaTeX operator rendering per transform *name* here (no `case` needed)
     */
    private val TRANSFORM_LATEX: Map [String, String] = Map ("pow"  -> "\\text{pow}",
                                                             "root" -> "\\text{root}",
                                                             "log"  -> "\\log",
                                                             "sin"  -> "\\sin",
                                                             "cos"  -> "\\cos")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Render a TransformT to a LaTeX operator using its .name.
     *  @param t  the transform type
     */
    private def latexOpFromTransform (t: TransformT): String =
        TRANSFORM_LATEX.getOrElse (t.name, s"\\text{${t.name}}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert a single feature token into a LaTeX term with coefficient.
     *  Supported tokens:
     *
     *    const
     *    ylK
     *    fI(ylK)
     *    xeJlK
     *    gJ,T(xeJlK)   (J = exo index, T = transform index for exo J)
     *
     *  @param feat           raw feature token (trimmed)
     *  @param fEndo_enabled  enabled transforms for y (order matters)
     *  @param fExo_enabled   enabled transforms per exo variable (order matters)
     *  @param coeffSym       e.g., "\\beta"
     *  @param termIdx        coefficient subscript index
     */
    private def feature2LatexTerm (feat: String, fEndo_enabled: LSET [TransformT],
                                   fExo_enabled: Array [LSET [TransformT]],
                                   coeffSym: String, termIdx: Int): String =
        // Regex patterns
        val R_CONST: Regex = raw"""^const$$""".r
        val R_YL:    Regex = raw"""^yl(\d+)$$""".r
        val R_FY:    Regex = raw"""^f\d+\(yl(\d+)\)$$""".r
    
        val R_XEL:   Regex = raw"""^xe(\d+)l(\d+)$$""".r
        val R_GX:    Regex = raw"""^g\d+,\d+\(xe(\d+)l(\d+)\)$$""".r
    
        def withCoeff (body: String): String = s"${coeffSym}_${termIdx}\\, $body"
    
        feat match
        case R_CONST() =>
            s"${coeffSym}_${termIdx}"
    
        case R_YL(k) =>
            val lag = k.toInt
            withCoeff (s"y_{t-$lag}")
    
        case R_FY(i, k) =>
            val idx = i.toInt
            val lag = k.toInt
            val endoArr = fEndo_enabled.toArray
            val op = if 0 <= idx && idx < endoArr.length
                    then latexOpFromTransform(endoArr(idx))
                    else s"\\text{f$idx}"
            withCoeff (s"$op\\big(y_{t-$lag}\\big)")
    
        case R_XEL(j, k) =>
            val exo = j.toInt
            val lag = k.toInt
            withCoeff (s"x^{(${exo})}_{t-$lag}")
    
        case R_GX(tIdx, j, k) =>
            // Trust the inner xe<j>l<k> indices.
            val exo  = j.toInt
            val lag  = k.toInt
            val trIx = tIdx.toInt
            val op =
            if 0 <= exo && exo < fExo_enabled.length then
                val exoArr = fExo_enabled(exo).toArray
                if 0 <= trIx && trIx < exoArr.length then latexOpFromTransform (exoArr(trIx))
                else s"\\text{g$trIx}"
            else s"\\text{g$trIx}"
            withCoeff (s"$op\\big(x^{(${exo})}_{t-$lag}\\big)")
    
        case _ =>
            // Fallback: still has coefficient, render raw token safely
            withCoeff (s"\\text{${feat.replace ("\\", "\\\\")}}")
    end feature2LatexTerm
        
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a LaTeX equation from feature tokens.
     *  @param features       the feature names/tokens (any order/subset)
     *  @param fEndo_enabled  enabled endo transforms (order matters)
     *  @param fExo_enabled   enabled exo transforms per variable (order matters)
     *  @param includeHat     whether to render \hat{y}_t on LHS
     *  @param coeffSymbol    coefficient symbol, e.g. "\\beta"
     *  @param errorSymbol    error term, e.g. "\\varepsilon_t"
     */
    def features2LatexEquation (features: Array [String], fEndo_enabled: LSET [TransformT],
                                fExo_enabled: Array [LSET [TransformT]], includeHat: Boolean = true,
                                coeffSymbol: String = "\\beta",
                                errorSymbol: String = "\\varepsilon_t"): String =
    
        val lhs   = if includeHat then "\\hat{y}_t" else "y_t"
        val terms = features.zipWithIndex.map { case (f, i) =>
                    feature2LatexTerm (f.trim, fEndo_enabled, fExo_enabled, coeffSymbol, i) }
        val eq    = s"$lhs \\,=\\, ${terms.mkString (" \\,+\\, ")} \\,+\\, $errorSymbol"
        eq.replace (raw"\\_", raw"\_")
    end features2LatexEquation

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make a LaTeX equation with the begin and end tags.
     *  @param equation  the LaTex equation proper
     */
    def make_equation (equation: String): String =
        s"\\begin{equation*}\n$equation\n\\end{equation*}"

    private val preamble: String = """
\documentclass{article}
\usepackage{amsmath} % For advanced math features like \text, \frac, etc.
\usepackage[utf8]{inputenc} % To handle various characters
\begin{document}
"""

    private val table_hdr: String = """\begin{table}[h]
\centering
"""

    private val table_tlr: String = """\end{tabular}
\end{table}
"""

    private val endln = " \\\\ \\hline \n"
    private val endl2 = " \\\\ \\hline \\hline \n"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make a LaTeX table from a matrix.
     *  @param caption  the caption for the table
     *  @param name     the table name
     *  @param x        the matrix of data
     *  @param colName  the column names
     *  @param rowName  the row names
     */
    def make_table (caption: String, name: String, x: MatrixD,
                    colName: String = null, rowName: Array [String] = null): String =
        var n  = x.dim2
        if rowName != null then n += 1
        val cs = "|c"*n + "|"
        val sb = new StringBuilder ()
        sb   ++= s"\\caption{$caption}\n"
        sb   ++= s"\\label{tab:$name}\n"
        sb   ++= s"\\begin{tabular}{$cs} \\hline \n"
        if colName != null then
            sb ++= colName.replace (",", " &") + endl2
        for i <- x.indices do
            if rowName != null then
                sb ++= rowName(i) + " &"
            sb ++= x(i).toString.replace ("VectorD(", "").replace (")", "").replace (",", " &") + endln
        s"$table_hdr${sb.mkString}$table_tlr"
    end make_table

    def make_body (): String =

        """
section {Auto MPG}
(398 x 7)
\url{https://archive.ics.uci.edu/dataset/9/auto+mpg}

section {House Price Regression Dataset}
(1000 x 8)
\url{https://www.kaggle.com/datasets/prokshitha/home-value-insights }

section {Each Group Picks a Dataset}
(more rows and columns)
        """

    end make_body

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make a LaTeX document.
     *  @param body  the body of the LaTex document
     */
    def make_doc (body: String): String =
        s"$preamble\n$body\n\\end{document}"
    end make_doc

end Scala2LaTeX


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `scala2LaTeXTest` main method tests the `Scala2LaTeX` object by making a LaTeX
 *  equation from a ScalaTion model specification.
 *  > runMain scalation.mathstat.scala2LaTeXTest
 */
@main def scala2LaTeXTest (): Unit =

    import Scala2LaTeX._

    val features = Array ("new\\_deaths", "icu\\_patients")
    val fEndo    = LSET [TransformT] ()
    val fExo     = Array [LSET [TransformT]] ()
    val latex    = make_doc (make_equation (features2LatexEquation (features, fEndo, fExo)))
    println (latex)

end scala2LaTeXTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `scala2LaTeXTest2` main method tests the `Scala2LaTeX` object by making a LaTeX
 *  table from a matrix.
 *  @see `scalation.modeling.Regression`
 *  > runMain scalation.mathstat.scala2LaTeXTest2
 */
@main def scala2LaTeXTest2 (): Unit =

    import Scala2LaTeX._

    // 16 data points:         one      x1      x2       x3     y
    //                         Const   Lat    Elev     Long  Temp        County
    val xy = MatrixD ((16, 5), 1.0, 29.767,   41.0,  95.367, 56.0,    // Harris
                               1.0, 32.850,  440.0,  96.850, 48.0,    // Dallas
                               1.0, 26.933,   25.0,  97.800, 60.0,    // Kennedy
                               1.0, 31.950, 2851.0, 102.183, 46.0,    // Midland
                               1.0, 34.800, 3840.0, 102.467, 38.0,    // Deaf Smith
                               1.0, 33.450, 1461.0,  99.633, 46.0,    // Knox
                               1.0, 28.700,  815.0, 100.483, 53.0,    // Maverick
                               1.0, 32.450, 2380.0, 100.533, 46.0,    // Nolan
                               1.0, 31.800, 3918.0, 106.400, 44.0,    // El Paso
                               1.0, 34.850, 2040.0, 100.217, 41.0,    // Collington
                               1.0, 30.867, 3000.0, 102.900, 47.0,    // Pecos
                               1.0, 36.350, 3693.0, 102.083, 36.0,    // Sherman
                               1.0, 30.300,  597.0,  97.700, 52.0,    // Travis
                               1.0, 26.900,  315.0,  99.283, 60.0,    // Zapata
                               1.0, 28.450,  459.0,  99.217, 56.0,    // Lasalle
                               1.0, 25.900,   19.0,  97.433, 62.0)    // Cameron

    val colName = "Const, Lat, Elev, Long, Temp"
    val caption = "Texas Temperatures Regression"
    val name    = "Texas-Temps"
    val latex   = make_doc (make_table (caption, name, xy, colName))
    println (latex)

end scala2LaTeXTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `scala2LaTeXTest3` main method tests the `Scala2LaTeX` object by making a LaTeX
 *  table from matrix of QoF metric vs. model.  This test does In-Sample testing.
 *  @see `scalation.modeling.Regression`
 *  > runMain scalation.mathstat.scala2LaTeXTest3
 */
@main def scala2LaTeXTest3 (): Unit =

    import Scala2LaTeX._

    // 16 data points:         one      x1      x2       x3     y
    //                         Const   Lat    Elev     Long  Temp        County
    val xy = MatrixD ((16, 5), 1.0, 29.767,   41.0,  95.367, 56.0,    // Harris
                               1.0, 32.850,  440.0,  96.850, 48.0,    // Dallas
                               1.0, 26.933,   25.0,  97.800, 60.0,    // Kennedy
                               1.0, 31.950, 2851.0, 102.183, 46.0,    // Midland
                               1.0, 34.800, 3840.0, 102.467, 38.0,    // Deaf Smith
                               1.0, 33.450, 1461.0,  99.633, 46.0,    // Knox
                               1.0, 28.700,  815.0, 100.483, 53.0,    // Maverick
                               1.0, 32.450, 2380.0, 100.533, 46.0,    // Nolan
                               1.0, 31.800, 3918.0, 106.400, 44.0,    // El Paso
                               1.0, 34.850, 2040.0, 100.217, 41.0,    // Collington
                               1.0, 30.867, 3000.0, 102.900, 47.0,    // Pecos
                               1.0, 36.350, 3693.0, 102.083, 36.0,    // Sherman
                               1.0, 30.300,  597.0,  97.700, 52.0,    // Travis
                               1.0, 26.900,  315.0,  99.283, 60.0,    // Zapata
                               1.0, 28.450,  459.0,  99.217, 56.0,    // Lasalle
                               1.0, 25.900,   19.0,  97.433, 62.0)    // Cameron

    val x   = xy.not (?, xy.dim2-1)                                   // matrix of predictor variable values
    val y   = xy(?, xy.dim2-1)                                        // original scale response vector
    val xy_ = xy(?, 1 until xy.dim2)                                  // drop the column of all ones from xy
    val x_  = x(?, 1 until x.dim2)                                    // drop the column of all ones from x

    val n_q = 15                                                      // number of core metrics
    val r_q = 0 until 15                                              // range of core metrics

    val colName = "Metric, Regression, Ridge, Lasso, Transformed, Symbolic"  // column names -- which models
    val rowName = modeling.qoF_names.take(n_q)                        // row names -- which QoF metric

    banner ("Regression Model")
    val mod1 = modeling.Regression (xy)()                             // Regression model
    val qof1 = mod1.trainNtest ()()._2(r_q)                           // train and test on full dataset
    println (mod1.summary ())

    banner ("Ridge Regression Model")
    val mod2 = modeling.RidgeRegression (xy_)()                       // Ridge Regression model
    val qof_ = mod2.trainNtest ()()._2(r_q)                           // train and test on full dataset
    val qof2 = modeling.RidgeRegression.fix_smape (mod2, y, qof_)
    println (mod2.summary ())

    banner ("Lasso Regression Model")
    modeling.RidgeRegression.hp("lambda") = 0.02                      // adjust shrinkage hyper-parameter
    val mod3 = modeling.LassoRegression (xy)()                        // Lasso Regression model
    val qof3 = mod3.trainNtest ()()._2(r_q)                           // train and test on full dataset
    println (mod3.summary ())

    banner ("Transfomed Regression Model")
    val mod4 = new modeling.TranRegression (x, y)                     // Transformed Regression model
                                                                      // defaults to log1p, try other transformations
    val qof4 = mod4.trainNtest ()()._2(r_q)                           // train and test on full dataset
    println (mod4.summary ())

    banner ("Symbolic Regression Model")
    val mod5 = modeling.SymRidgeRegression.quadratic (x_, y)          // simple Symbolic Regression model
                                                                      // try other forms for Symbolic Regression
    val qof5 = mod5.trainNtest ()()._2(r_q)                           // train and test on full dataset
    println (mod5.summary ())

    banner ("Copy-paste the LateX table below into a .tex file")

    val qofs = MatrixD (qof1, qof2, qof3, qof4, qof5).transpose       // put QoFs in a matrix

    val caption = "Texas Temperatures: Regression, Ridge, Lasso, Transformed"   // LaTex figure caption
    val name    = "Texas-Temps"

    val latex   = make_doc (make_table (caption, name, qofs, colName, rowName))
    println (latex)

end scala2LaTeXTest3

