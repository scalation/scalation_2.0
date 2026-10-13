
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Aug 26 13:42:17 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Tool: Periodogram Power vs. ?
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Periodogram` class is used to analyze the frequency spectrum of a time series.
 *  @param y       the given time series vector
 *  @param name    the namer of the time series vector
 *  @param center  whether the center the data before taking FFT
 */
class Periodogram (y: VectorD, name: String, center: Boolean = true):

    private val n    = y.dim                                              // size of time series
    private val half = n / 2                                              // half size
    private val y_   = if center then y - y.mean else y                   // should center (or even detrend)
    private val x    = VectorC.fromDoubles (y_)                           // convert to complex numbers

    calculus.FFT.fft (x)                                                  // converts x into frequencies using FFT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the single-sided amplitude vector (Magnitude spectrum).
     */
    def amplitudes: VectorD = 
        val amps = new VectorD (half)
        for i <- 1 until half do amps(i) = x(i).mag
        amps
    end amplitudes

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Extract the top k unique frequencies with the highest spectral power/amplitude.
     *  @param amps  the vector of amplitudes
     *  @param k     the number of peak periods to return
     */
    def topKFrequencies (amps: VectorD, k: Int = 10): VectorD =
        val totalSamples = amps.length * 2
    
        VectorD (amps.zipWithIndex
            .drop (1)              // skip index 0 (DC offset/mean) to ensure we only capture actual oscillations
            .sortBy (-_._1)        // sort descending based on the amplitude magnitude value
            .take (k)              // slice the top K strongest signals
            .map { case (_, index) => index.toDouble / totalSamples })  // convert index back to frequency: f = index / N
    end topKFrequencies

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Extract the top k unique seasonal periods by inverting the top k discovered frequencies (P = 1 / f).
     *  @param amps  the vector of amplitudes
     *  @param k     the number of peak periods to return
     */
    def topKPeriods (amps: VectorD, k: Int = 10): VectorD =
        topKFrequencies (amps, k).map (f => 1.0 / f)
    end topKPeriods

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Plot the frequency domain signal.
     */
    def showPlot (): Unit =
        new Plot (null, x.toDouble, null, s"frequency domain signal for $name", lines = true)
    end showPlot

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Plot Amplitude vs. Frequency (f) where frequency ranges linearly from 0 to 0.5 cycles per step.
     *  @param amps  the vector of amplitudes
     */
    def plotVsFrequency (amps: VectorD): Unit =
        val freqX        = new VectorD (amplitudes.length)
        val totalSamples = amplitudes.length * 2
        
        // Map linear frequency coordinates to the X axis vector
        for i <- amps.indices do
            freqX(i) = i.toDouble / totalSamples
            
        new Plot (freqX, amplitudes, null, s"$name Periodogram: Amplitude vs. Frequency", lines = true)
    end plotVsFrequency

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Plots Amplitude vs. Time Period (P) where Period is non-linear (P = Total_Samples / bin_index).
     *  @param amps  the vector of amplitudes
     */
    def plotVsPeriod (amps: VectorD): Unit =
        // We drop index 0 (infinity period) to prevent divide-by-zero layout errors
        val validSize    = amplitudes.length - 1
        val periodX      = new VectorD (validSize)
        val filteredY    = new VectorD (validSize)
        val totalSamples = amplitudes.length * 2
        
        for i <- 1 until amps.length do
            periodX(i - 1)   = totalSamples.toDouble / i
            filteredY(i - 1) = amplitudes(i)
            
        new Plot (periodX, filteredY, null, s"$name Periodogram: Amplitude vs. Time Period", lines = true)
    end plotVsPeriod

end Periodogram


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `periodogramTest` main function tests the `Periodogram` class on real data.
 *  > runMain scalation.modeling.forecasting.periodogramTest
 */
@main def periodogramTest (): Unit =

    import Example_Covid.{clip, loadData}

    val vars    = Array ("icu_patients")
    val (xe, y) = clip (loadData (vars))
    val z       = xe(?, 0)

    new Plot (null, y, null, "new_deaths: endo time series y", lines = true)
    new Plot (null, z, null, "icu_patients: exo time series z", lines = true)

    var pg   = new Periodogram (y, "new_deaths")                          // apply Periodogram to column y
    var amps = pg.amplitudes
    pg.showPlot ()
    pg.plotVsFrequency (amps)
    pg.plotVsPeriod (amps)

    banner ("Top k frequencies and periods for new_deaths")
    println (s"frequencies = ${pg.topKFrequencies (amps)}")
    println (s"periods     = ${pg.topKPeriods (amps)}")

    pg   = new Periodogram (z, "icu_patients")                            // apply Periodogram to column z
    amps = pg.amplitudes
    pg.showPlot ()
    pg.plotVsFrequency (amps)
    pg.plotVsPeriod (amps)

    banner ("Top k frequencies and periods for icu_patients")
    println (s"frequencies = ${pg.topKFrequencies (amps)}")
    println (s"periods     = ${pg.topKPeriods (amps)}")

end periodogramTest

