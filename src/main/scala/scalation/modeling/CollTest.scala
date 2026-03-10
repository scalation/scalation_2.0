
package scalation
package modeling

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `collTest` main method test for collinearity between column vectors x1 and x2.
 *  > runMain scalation.modeling.collTest
 */
@main def collTest (): Unit =
//                         one x1 x2
    val x = MatrixD ((4, 3), 1, 1, 1,
                             1, 2, 2,
                             1, 3, 3,
                             1, 4, 0)                  // change 0 by .5 to 4
    val y = VectorD (1, 3, 3, 4)

//  Regression.hp("factorization") = "Fac_SVD"         // uncomment for singular matrix

    for _ <- 0 to 8 do
        banner (s"Test Increasing Collinearity: x_32 = ${x(3, 2)}")
        println (s"x = $x")
        println (s"x.corr = ${x.corr}")
        val mod = new Regression (x, y)
        mod.trainNtest ()()
        println (mod.summary ())
        x(3, 2) += 0.5

end collTest

