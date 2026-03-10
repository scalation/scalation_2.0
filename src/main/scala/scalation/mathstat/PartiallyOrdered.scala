
package scalation
package mathstat

// https://docs.scala-lang.org/contribute/bug-reporting-guide.html
// trait copied from Scala 3.7.0 API
// https://www.scala-lang.org/api/3.7.0/scala/math/PartiallyOrdered.html
// code for testing "unused implicit parameter" warning when using  "-Wunused:all",

trait PartiallyOrdered[+A] extends Any:

  type AsPartiallyOrdered[B] = B => PartiallyOrdered[B]

  /** Result of comparing `'''this'''` with operand `that`.
   *  Returns `None` if operands are not comparable.
   *  If operands are comparable, returns `Some(x)` where
   *  - `x < 0`    iff   `'''this''' &lt; that`
   *  - `x == 0`   iff   `'''this''' == that`
   *  - `x > 0`    iff   `'''this''' &gt; that`
   */
  infix def tryCompareTo [B >: A: AsPartiallyOrdered](that: B): Option[Int]

  def < [B >: A: AsPartiallyOrdered](that: B): Boolean =
    (this tryCompareTo that) match
      case Some(x) if x < 0 => true
      case _ => false

  def > [B >: A: AsPartiallyOrdered](that: B): Boolean =
    (this tryCompareTo that) match
      case Some(x) if x > 0 => true
      case _ => false

  def <= [B >: A: AsPartiallyOrdered](that: B): Boolean =
    (this tryCompareTo that) match
      case Some(x) if x <= 0 => true
      case _ => false

  def >= [B >: A: AsPartiallyOrdered](that: B): Boolean =
    (this tryCompareTo that) match
      case Some(x) if x >= 0 => true
      case _ => false


case class MyPartialOrder (x: Int, y: Int) extends PartiallyOrdered [MyPartialOrder]:

  infix def tryCompareTo [B >: MyPartialOrder: AsPartiallyOrdered](that: B): Option[Int] =

      that match
      case that: MyPartialOrder if x == that.x && y == that.y => Some (0)
      case that: MyPartialOrder if x <= that.x && y <= that.y => Some (-1)
      case that: MyPartialOrder if x >= that.x && y >= that.y => Some (1)
      case _ => None

/*
      if ! that.isInstanceOf [MyPartialOrder] then return None
      val b = that.asInstanceOf [MyPartialOrder]
      if x == b.x && y == b.y then      Some (0)
      else if x <= b.x && y <= b.y then Some (-1)
      else if x >= b.x && y >= b.y then Some (1)
      else None

[warn] -- [E198] Unused Symbol Warning: scalation_2.0/src/main/scala/scalation/mathstat/PartiallyOrdered.scala:41:65 
[warn] 41 |  infix def tryCompareTo [B >: MyPartialOrder: AsPartiallyOrdered](that: B): Option[Int] =
[warn]    |                                                                 ^
[warn]    |                                               unused implicit parameter

This there another solution besides using @nowarn (@unused did not work)?
For version 3.7.0, adding an explicit return, makes the warning go away.  This should be fixed in 3.7.1
Fixed by 3.7.1-RC2
*/

