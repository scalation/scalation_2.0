
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo, John Miller
 *  @version 2.0
 *  @date    Sat Aug 29 14:44:47 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Support for a Minimal Common API for Vectors and Matrices
 */

package scalation
package mathstat

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VecMat` type alias is shorthand for the union of `VectorD` and `MatrixD`.
 */
type VecMat = VectorD | MatrixD


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Defines several extension methods that work for either vectors or matrices, i.e.,
 *  for types that are either `VectorD` or `MatrixD`.
 *  @param x  the argument that is either a `VectorD` or `MatrixD`.
 */
extension (x: VecMat)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the size of the first dimension of the vector or matrix.
     */
    def dim: Int =
        x match
            case xV: VectorD => xV.dim
            case xM: MatrixD => xM.dim

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return mean and standard deviation of the vector or matrix.
     */
    def mu_sig: VecMat =
        x match
            case xV: VectorD => xV.mu_sig
            case xM: MatrixD => xM.mu_sig

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return minumim and maximum of the vector or matrix.
     */
    def min_max: VecMat =
        x match
            case xV: VectorD => xV.min_max
            case xM: MatrixD => xM.min_max

