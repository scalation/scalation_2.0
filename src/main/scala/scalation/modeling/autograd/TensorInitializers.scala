
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Fri April 25 19:56:12 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: Tensor Initialization Methods for Neural Networks
 */

package scalation
package modeling
package autograd

import scalation.mathstat.{MatrixD, TensorD}
import scalation.random.{NormalMat, RandomMatD}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TensorInitializers` utility object for tensor initializations commonly used in neural networks.
 *  Provides methods to create tensors filled with zeros, ones, random values,
 *  and standardized initialization schemes like He and Xavier initialization.
 *  All returned tensors have batch-first shape: (batch, rows, cols).
 */
object TensorInitializers:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a tensor of zeros with shape (batch, rows, cols).
     */
    def zeros (batch: Int = 1, rows: Int, cols: Int): TensorD = TensorD.fill (batch, rows, cols, 0.0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a tensor of ones with shape (batch, rows, cols).
     */
    def ones (batch: Int = 1, rows: Int, cols: Int): TensorD = TensorD.fill (batch, rows, cols, 1.0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert a MatrixD to a TensorD with shape (1, rows, cols).
     */
    def fromMatrix (m: MatrixD): TensorD = TensorD.fromMatrix (m)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Stack a sequence of matrices into a TensorD with batch dimension.
     *  Each matrix becomes one slice: resulting shape = (batch, rows, cols).
     */
    def fromMatrices (mats: IndexedSeq [MatrixD]): TensorD = TensorD (mats)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a tensor with random values from a uniform distribution N(0, 1).
     */
    def rand (batch: Int = 1, rows: Int, cols: Int): TensorD =
        val mats = for _ <- 0 until batch yield RandomMatD(rows, cols, 0.0, 1.0).gen
        fromMatrices (mats)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a tensor with random values from a normal distribution N(0, stdDev&#94;2).
     */
    def randn (batch: Int = 1, rows:  Int, cols:  Int, stdDev: Double = 1.0): TensorD =
        val mats = for _ <- 0 until batch yield NormalMat (rows, cols, 0.0, stdDev).gen
        fromMatrices (mats)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** He initialization (Kaiming initialization).
     *  Standard deviation: sqrt(2 / fanIn), where fanIn = number of input features.
     */
    def heInit (batch: Int = 1, rows: Int, cols: Int): TensorD =
        val std = math.sqrt(2.0 / rows)
        randn (batch, rows, cols, std)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Xavier initialization (Glorot initialization).
     *  Standard deviation: sqrt(2 / (fanIn + fanOut)).
     */
    def xavierInit (batch: Int = 1, rows: Int, cols: Int): TensorD =
        val std = math.sqrt (2.0 / (rows + cols))
        randn (batch, rows, cols, std)

// Additional initializers

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Uniform initializer: samples values from U(-bound, bound).
     *  @param batch number of tensor slices
     *  @param rows  number of rows per slice
     *  @param cols  number of columns per slice
     *  @param bound half-width of uniform interval
     *  @return a TensorD with uniform random values in [-bound, bound]
     */
    def uniform (batch: Int = 1, rows: Int, cols: Int, bound: Double): TensorD =
        val mats = for _ <- 0 until batch yield RandomMatD (rows, cols, bound, -bound).gen
        fromMatrices (mats)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Xavier/Glorot uniform initialization.
     *  Samples from U(± sqrt(6 / (fanIn + fanOut))).
     *  @param batch  the number of tensor slices (default 1)
     *  @param fanOut the number of output features (rows)
     *  @param fanIn  the number of input features (cols)
     *  @return a TensorD initialized using Xavier uniform initialization
     */
    def xavierUniform (batch: Int = 1, fanOut: Int, fanIn: Int): TensorD =
        val bound = math.sqrt (6.0 / (fanIn + fanOut))
        uniform (batch, fanOut, fanIn, bound)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** LeCun uniform initialization.
     *  Samples from U(± sqrt(3 / fanIn)).  Suitable for tanh activations.
     *  @param batch  the number of tensor slices (default 1)
     *  @param fanOut the number of output features (rows)
     *  @param fanIn  the number of input features (cols)
     *  @return a TensorD initialized using LeCun uniform initialization
     */
    def lecunUniform (batch: Int = 1, fanOut: Int, fanIn: Int): TensorD =
        val bound = math.sqrt (3.0 / fanIn)
        uniform (batch, fanOut, fanIn, bound)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** PyTorch-style RNN parameter initialization.
     *  Samples from U(± 1/sqrt(hiddenSize)) for all recurrent parameters.
     *  @param batch      the number of tensor slices (default 1)
     *  @param hiddenSize size of the hidden state used to compute bound
     *  @param rows       the number of rows per slice
     *  @param cols       the number of columns per slice
     *  @return a TensorD initialized using RNN uniform initialization
     */
    def rnnUniform (batch: Int = 1, hiddenSize: Int, rows: Int, cols: Int): TensorD =
        val bound = 1.0 / math.sqrt (hiddenSize.toDouble)
        uniform (batch, rows, cols, bound)

end TensorInitializers

