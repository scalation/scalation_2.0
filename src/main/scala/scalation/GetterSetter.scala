
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Oct  1 13:01:06 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Illustration of Creation and Use of Getter and Setter Methods in Scala 3
 *
 *  @see     docs.scala-lang.org/tour/classes.html
 */

package scalation

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GetterSetter` object provides an example of how to create getter and setter methods
 *  in Scala 3: private field variable '_name', getter method 'name', setter method 'name_='.
 *  @note: When a field is val (not var), it is reasonable to make it public as its value can
 *  be accessed, but not changed, i.e., custom getter and setter methods need not be written.
 *  @see kkyr.io/blog/getters-and-setters-in-scala
 */
object GetterSetter:                                                         // for object, trait or class

    private val flaw   = flawf ("GetterSetter")                              // ScalaTion convention for error messages
    private var _width = 1.2                                                 // example private variable

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the value of the _width` variable.  May use inline for efficiency.
     */
    inline def width: Double = _width

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the variable `_width` to the new value `width_` checking its validity.
     *  If the error/flaw is severe, may set `_width` to a default value or throw an exception.
     *  @param width_  the new width to be assigned
     */
    def width_= (width_ : Double): Unit =
        if width_ < 0.0 then flaw ("width_=", s"width_ = $width_ but should be non-negative")
        _width = width_
    end width_=

end GetterSetter

 
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SetterOnly` class provides an example of how to create a setter method for a protected
 *  variable.
 */
class SetterOnly ():                                                         // for trait or class

    private val flaw     = flawf ("SetterOnly")                              // ScalaTion convention for error messages
    protected var length = 1.2                                               // variable only accessible in hierarchy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the variable `length` to the new value `length_` checking its validity.
     *  If the error/flaw is severe, may set `length` to a default value or throw an exception.
     *  @param length_  the new length to be assigned
     */
    def setLength (length_ : Double): Unit =
        if length_ < 0.0 then flaw ("setLength", s"length_ = $length_ but should be non-negative")
        length = length_
    end setLength

end SetterOnly


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `getterSetterTest` main function shows how to use getter and setter methods in Scala 3.
 *  > runMain scalation.getterSetterTest
 */
@main def getterSetterTest (): Unit =

    import GetterSetter._

    banner ("Test Custom GetterSetter object")
    println (s"the current width = $width")                                  // call to getter
    width = 2.3                                                              // call to setter
    println (s"the updated width = $width")
    width = -3.4
    println (s"the final width   = $width")

    banner ("Test SetterOnly class")
    val prop = SetterOnly ()
    prop.setLength (2.3)                                                     // call to setter
    prop.setLength (-3.4)                                                    // call to setter

end getterSetterTest

