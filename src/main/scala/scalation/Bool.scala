
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun May  7 18:20:18 EDT 2023 
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Extensions Method for `Boolean`: and, or, not, xor
 */

package scalation

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Extend `Boolean` to include an and, or, not, xor.
 *  @note:  unfortunately they all have the same precedence.
 */
extension (x: Boolean)
    inline infix def and (y: Boolean): Boolean = x && y
    inline infix def or (y: Boolean): Boolean  = x || y
    inline infix def xor (y: Boolean): Boolean = x != y
    inline infix def not: Boolean = ! x


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `boolTest` main function tests the `Bool` extension methods.
 *  > runMain scalation.boolTest
 */
@main def boolTest (): Unit =

    banner ("Compare && with and")
    for i <- 1 to 3; j <- 1 to 3  do
        println (s"i < j && i > 1  for i = $i, j = $j is ${i < j && i > 1}")
        println (s"i < j and i > 1 for i = $i, j = $j is ${i < j and i > 1}")

    banner ("Compare || with or")
    for i <- 1 to 3; j <- 1 to 3  do
        println (s"i < j || i > 1 for i = $i, j = $j is ${i < j || i > 1}")
        println (s"i < j or i > 1 for i = $i, j = $j is ${i < j or i > 1}")

    banner ("Compare != with xor")
    for i <- 1 to 3; j <- 1 to 3  do
        println (s"i < j != i > 1  for i = $i, j = $j is ${i < j != i > 1}")
        println (s"i < j xor i > 1 for i = $i, j = $j is ${i < j xor i > 1}")

    banner ("Compare ! with not")
    for i <- 1 to 3; j <- 1 to 3  do
        println (s"! i < j   for i = $i, j = $j is ${! (i < j)}")
        println (s"not i < j for i = $i, j = $j is ${not (i < j)}")

end boolTest

