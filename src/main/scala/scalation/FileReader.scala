
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Aug 28 18:17:12 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Top-Level Functions to Read from Files (FileReader) in ScalaTion's
 *           "data" Directory or from a full-path
 */

package scalation

import java.io.IOException

import scala.collection.mutable.ArrayBuffer
import scala.io.{BufferedSource, Source}

val _flaw = flawf ("FileReader")                                       // flaw function

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Read and print each line in the file.
 *  @see `DATA_DIR` in Util.scala
 *  @param fileName  the file name (relative or full path)
 *  @param fullPath  flag indicating whether the user wants full or relative file paths
 *                   defaults to false (relative paths)
 */
def readFile (fileName: String, fullPath: Boolean = false): Int =
    val path = if fullPath then fileName
               else DATA_DIR + fileName                                // relative to DATA_DIR
    println (s"readFile: $path")
    var buffer: BufferedSource = null
    try
        buffer = Source.fromFile (path)                                // @see BufferedSource
    catch 
        case _ : IOException => _flaw ("readFile", s"IOException: file $path may not exist.")

    val lines  = buffer.getLines
    var i = 0
    for line <- lines do
        println (line)
        i += 1
    end for
    buffer.close ()
    i
end readFile


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Read each line in the file and put each one into an array that's returned.
 *  @see `DATA_DIR` in Util.scala
 *  @see `getFromURL_File` in Util.scala has similar functionality, but returns an `Iterator`
 *  @param fileName  the file name (relative or full path)
 *  @param fullPath  flag indicating whether the user wants full or relative file paths
 *                   defaults to false (relative paths)
 *  @param limit     the limit on the number of lines to read
 */
def readFileIntoArray (fileName: String, fullPath: Boolean = false, limit: Int = -1): Array [String] =
    val path = if fullPath then fileName
               else DATA_DIR + fileName                                // relative to DATA_DIR
    println (s"readFileIntoArray: $path")
    var buffer: BufferedSource = null
    try
        buffer = Source.fromFile (path)                                // @see BufferedSource
    catch 
        case _ : IOException => _flaw ("readFileIntoArray", s"IOException: file $path may not exist.")

    val lineArr = 
    if limit <= 0 then
        buffer.getLines.toArray
    else
        val it: Iterator [String] = buffer.getLines ()                 // line iterator
        val ab = ArrayBuffer [String] ()
        var i = 0
        while i < limit && it.hasNext do
            ab += it.next ()
            i  += 1
        end while
        ab.toArray
    end lineArr

    buffer.close
    println (s"readFileIntoArray: number lines = ${lineArr.size}")     // check number of lines
    println (s"readFileIntoArray: lines 0 = ${lineArr (0)}")           // and first two lines
    println (s"readFileIntoArray: lines 1 = ${lineArr (1)}")
    lineArr
end readFileIntoArray


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Read the file one line at a time by returning an iterator.  The client code
 *  should call hasNext to see if there are more lines, next to read the next line,
 *  and close to close the buffer at the end.  The iterator and buffer are returned.
 *  @see `DATA_DIR` in Util.scala
 *  @param fileName  the file name (relative or full path)
 *  @param fullPath  flag indicating whether the user wants full or relative file paths
 *                   defaults to false (relative paths)
 */
def readFileIter (fileName: String, fullPath: Boolean = false): (Iterator [String], BufferedSource) =
    val path = if fullPath then fileName
               else DATA_DIR + fileName                                // relative to DATA_DIR
    println (s"readFileIterator: $path")
    var buffer: BufferedSource = null
    try
        buffer = Source.fromFile (path)                                // @see BufferedSource
    catch 
        case _ : IOException => _flaw ("readFileIter", s"IOException: file $path may not exist.")

    val it: Iterator [String] = buffer.getLines ()                     // line iterator
    (it, buffer)                                                       // return iterator and buffer
end readFileIter


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `readFileTest` main function is used to test the file reading functions.
 *  Read the travel time dataset.
 *  > runMain scalation.readFileTest
 */
@main def readFileTest (): Unit =

    val fileName = "travelTime.csv"

    banner (s"readFile ($fileName)")
    println (s"readFile: number of lines = ${readFile (fileName)}")

    banner (s"readFileIntoArray ($fileName)")
    val lineArr = readFileIntoArray (fileName)
    println (s"readFileIntoArray: number of lines = ${lineArr.length}")
 
end readFileTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `readFileTest2` main function is used to test the file reading functions.
 *  Read the COVID-19 dataset with a limit of 200 lines.
 *  > runMain scalation.readFileTest2
 */
@main def readFileTest2 (): Unit =

    val fileName = "covid_19.csv"

    banner (s"readFile ($fileName)")
    println (s"readFile: number of lines = ${readFile (fileName)}")

    banner (s"readFileIntoArray ($fileName, limit = 200)")
    val lineArr = readFileIntoArray (fileName, limit = 200)
    println (s"readFileIntoArray: number of lines = ${lineArr.length}")
 
end readFileTest2

