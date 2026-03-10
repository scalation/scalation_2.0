
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Dec 21 12:53:44 EST 2019
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Neural Network with 4+ Layers Supporting Transfer Learning
 */

package scalation
package modeling
package neuralnet

import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

import ActivationFun._
import Initializer._
import Optimizer._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_XLT` class supports multi-output, multi-layer (input, {hidden} and output)
 *  Neural-Networks with Transfer Learning.  A layer (first hidden by default) from a neural-
 *  network model trained on a related dataset is transferred into that position in this model.
 *  Given several input vectors and output vectors (training data), fit the parameters b
 *  connecting the layers, so that for a new input vector v, the net can predict the output vector.
 *  Caveat: currently only allows the transfer of one layer.
 *  @param x         the m-by-n input matrix (training data consisting of m input vectors)
 *  @param y         the m-by-ny output matrix (training data consisting of m output vectors)
 *  @param fname_    the feature/variable names (defaults to null)
 *  @param nz        the number of nodes in each hidden layer, e.g., Array (9, 8) => 2 hidden of sizes 9 and 8 
 *                   (null => use default formula)
 *  @param hparam    the hyper-parameters for the model/network
 *  @param f         the array of activation function families between every pair of layers
 *  @param l_tran    the layer to be transferred in (defaults to first hidden layer)
 *  @param transfer  the saved network parameters from a layer of a related neural network
 *                   trim before passing in if the size does not match
 *  @param itran     the inverse transformation function returns responses to original scale
 */
class NeuralNet_XLT (x: MatrixD, y: MatrixD, fname_ : Array [String] = null,
                     nz: Array [Int] = null, hparam: HyperParameter = hp,
                     f: Array [AFF] = Array (f_sigmoid, f_sigmoid, f_id),
                     l_tran: Int = 1, transfer: NetParam = null,
                     itran: FunctionM2M = null)
      extends NeuralNet_XL (x, y, fname_, nz, hparam, f, itran):

    bb = (for l <- layers yield 
              if l == l_tran && transfer != null then transfer                // replace `NetParam`
              else new NetParam (weightMat (sizes(l), sizes(l+1)),            // parameters weights &
                                 weightVec (sizes(l+1)))).toArray             // biases per active layer

    _modelName = s"NeuralNet_XLT_${stringOf (f.map (_.name))}"

    println (s"Create a NeuralNet_XLT with ${x.dim2} input, ${stringOf (nz)} hidden and ${y.dim2} output nodes")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Trim the new transferred in `NetParam` to match the size of the existing `NetParam`.
     *  @param tl  the transferred in layer's NetParam (weights and biases) (defaults to the l_tran layer)
     *  @param lt  the layer to transfer in
     */
    def trim (tl: NetParam, lt: Int = l_tran): NetParam = tl.trim (sizes(lt), sizes(lt+1))

end NeuralNet_XLT


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_XLT` companion object provides factory methods for creating multi-layer
 *  (one+ hidden layers) neural networks supporting transfer learning.  Note, 'scale' is
 *  defined in `Scaling`.
 */
object NeuralNet_XLT extends Scaling:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NeuralNet_XLT` for a combined data matrix.
     *  @param xy        the combined input and output matrix
     *  @param fname     the feature/variable names
     *  @param nz        the number of nodes in each hidden layer, e.g.,
     *                   Array (5, 10) means 2 hidden with sizes 5 and 10
     *  @param hparam    the hyper-parameters
     *  @param f         the array of activation function families over all layers
     *  @param l_tran    the layer to be transferred in (defaults to first hidden layer)
     *  @param transfer  the saved network parameters from a layer of a related neural network
     *  @param col       the first designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               nz: Array [Int], hparam: HyperParameter = hp,
               f: Array [AFF] = Array (f_sigmoid, f_sigmoid, f_id),
               l_tran: Int = 1, transfer: NetParam = null)
              (col: Int = xy.dim2 - 1): NeuralNet_XLT =
        var itran: FunctionM2M = null                                         // inverse transform -> original scale
        val (x, y) = (xy(?, 0 until col), xy(?, col until xy.dim2))           // assumes the last column is the response

        val x_s = if scale then rescaleX (x, f(0))
                  else x
        val y_s = if f.last.bounds != null then { val y_i = rescaleY (y, f.last); itran = y_i._2; y_i._1 }
                  else y

//      println (s" scaled: x = $x_s \n scaled y = $y_s")
        new NeuralNet_XLT (x_s, y_s, fname, nz, hparam, f, l_tran, transfer, itran)
    end apply

end NeuralNet_XLT

import Example_AutoMPG._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_XLTTest` object trains a neural network on the `Example_AutoMPG` dataset.
 *  This test case does not use transfer learning.
 *  A Neural Network with 2 hidden layers is created.
 *  > runMain scalation.modeling.neuralnet.neuralNet_XLTTest
 */
@main def neuralNet_XLTTest (): Unit =

    banner ("NeuralNet_XLT - no transfer learning - Example_AutoMPG")

    banner ("NeuralNet_XLT with scaled y values")
//  hp("eta") = 0.0014                                              // try several values - train
    hp("eta") = 0.02                                                // try several values - train2

    val nz = Array (7, 5)                                           // sizes for two hidden layers
    val nn = NeuralNet_XLT (xy, x_fname, nz)()                      // factory method automatically rescales
//  val nn = new NeuralNet_XLT (x, MatrixD.fromVector (y))          // constructor does not automatically rescale

    nn.trainNtest ()()                                              // fit the weights using training data (0, 1, 2)

//  banner ("cross-validation")
//  nn.crossValidate ()

end neuralNet_XLTTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_XLTTest2` object trains a neural network on the `Example_AutoMPG` dataset.
 *  This test case uses transfer learning.
 *  A Neural Network with 2 hidden layers is created with the first hidden layer being
 *  transferred from a model trained on related data.
 *  FIX: find a related dataset for 'nn0'.
 *  > runMain scalation.modeling.neuralnet.neuralNet_XLTTest2
 */
@main def neuralNet_XLTTest2 (): Unit =

    banner ("NeuralNet_XLT - transfer learning - Example_AutoMPG")

    banner ("NeuralNet_XLT - train using related data")
    hp("eta") = 0.02                                                // try several values - train

    val nz0 = Array (7, 5)                                          // sizes for two hidden layers
    val nn0 = NeuralNet_XLT (xy, x_fname, nz0)()                    // factory method automatically rescales
//  val nn0 = new NeuralNet_XLT (x, MatrixD.fromVector (y))         // constructor does not automatically rescale

    nn0.trainNtest ()()                                             // fit the weights using training data (0, 1, 2)
    val b1 = nn0.getNetParam ()                                     // get the parameters (weights and biases) from the first hidden layer

    banner ("NeuralNet_XLT with scaled y values")
//  hp("eta") = 0.0014                                              // try several values - train
    hp("eta") = 0.02                                                // try several values - train2

    val nz = Array (7, 5)                                           // sizes for two hidden layers
    val nn = NeuralNet_XLT (xy, x_fname, nz, transfer = b1)()       // factory method automatically rescales, transfer b1
//  val nn = new NeuralNet_XLT (x, MatrixD.fromVector (y))          // constructor does not automatically rescale

    nn.trainNtest ( )()                                             // fit the weights using training data (0, 1, 2)

//  banner ("cross-validation")
//  nn.crossValidate ()

end neuralNet_XLTTest2

