
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Srikalyan Swayampakula, Michael E. Cotterell
 *  @version 2.0
 *  @date    Wed Aug  27 14:16:12 EDT 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Utility to Generate index.html Files for Traversing ScalaTion Source Code
 */

package scalation

import java.io.{BufferedWriter, File, FileWriter}

import scala.collection.mutable.ArrayBuffer

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `genIndexHtml` main function is used to create "index.html" files in source code
 *  directories (main and test) to enable Web browsing of source code.
 *  > runMain scalation.GenIndexHtml
 */
@main def genIndexHtml (): Unit =

    val SKIP = "old"                     // do not process files in this directory

    val mainCodeDir = SRC_DIR
    println ("Generate index.html files starting from mainCodeDir = " + mainCodeDir)
    recCreate (new File (mainCodeDir))

/***
    val mainCodeDir = SRC_DIR + "main" + ⁄ + "scala"
    val testCodeDir = SRC_DIR + "test" + ⁄ + "scala"
    println ("Generate index.html files starting from testCodeDir = " + testCodeDir)
    recCreate (new File (testCodeDir))
***/

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursively create index.html files for each directory.
     *  @param f  the file/directory to examine
     */
    def recCreate (f: File): Unit =
        recDeleteIndex (f)
        val dirs = new ArrayBuffer [File] ()

        try
            val iFile = new File (f.getAbsolutePath () + "/index.html")      // the index.html file to write
            val fos   = new BufferedWriter (new FileWriter (iFile))
            fos.write ("<html>\n<body>\n<h1> Source files in " + f.getName () + " Package </h1><p>\n<ul>\n")

            for fi <- f.listFiles () sortWith ( (f1, f2) => f1.getName.toLowerCase < f2.getName.toLowerCase ) do
                val fName = fi.getName ()
                if ! fi.isDirectory () && fName != "index.html" then
                    fos.write ("<li> <a href = './" + fName + "'> " + fName + " </a> </li>\n")
                else if fName != SKIP && fName != "index.html" then
                    dirs += fi
            end for

            for fi <- dirs do
                val fName = fi.getName ()
                if fName != SKIP then
                    fos.write ("<li> <a href = './" + fName + "'> " + fName + " </a> </li>\n")
            end for

            fos.write ("</ul>\n</body>\n<html>")
            fos.close ()

            for fi <- dirs if fi.isDirectory () do recCreate (fi)     // recurse into each directory
        catch
            case _ : Throwable =>

        end try
    end recCreate

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursively delete index.html files for each directory (clean up step).
     *  @param f  the file/directory to examine
     */
    def recDeleteIndex (f: File): Unit =
        if ! f.isDirectory () then
            if f.getName () == "index.html" then f.delete ()
        else
            val files = f.listFiles ()
            if files != null then
                for fi <- files do try recDeleteIndex (fi) catch { case _ : Throwable => }
        end if
    end recDeleteIndex

end genIndexHtml

