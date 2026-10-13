
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Aug 29 15:30:35 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: For Model that Require No Training
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NoTrainModel` trait is for models like `RandomWalk`, which require no training.
 */
trait NoTrainModel:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Some models do not require training/fitting of parameters, this train method
     *  does nothing.
     *  @param x_  the data/input matrix
     *  @param y_  the testing/full response/output vector
     */
    def train (x_ : MatrixD, y_ : VectorD): Unit = {}

end NoTrainModel

