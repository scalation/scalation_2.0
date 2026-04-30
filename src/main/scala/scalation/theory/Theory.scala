
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Sep 28 18:02:55 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Class Bringing Together the Elements of Modeling and Theory
 */

package scalation
package theory

import scala.collection.mutable.{LinkedHashSet => LSET}
import scala.runtime.ScalaRunTime.stringOf

import scalation.database.table.LTable
import scalation.mathstat._
import scalation.modeling._
import scalation.modeling.forecasting.{Forecaster, Forecaster_Reg}

import modeling.newFname

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Dataset` class supports loading data files (e.g., CSV files) and pre-processing
 *  them to create predictor/input matrices and response/output vectors.
 *  FIX -- extend to models that allow multiple responses/outputs (vector -> matrix).
 *  @param name      the name of the dataset
 *  @param fileName  the name of the file storing the dataset 
 *  @parm  ncols     the number of columns in the dataset
 *  @param xcols     the desired column numbers to take for predictors
 *  @param ycol      the desired column number to take for the response
 */
case class Dataset (name: String, fileName: String, ncols: Int, xcols: Array [Int], ycol: Int):

    val rawData = LTable.load (fileName, name, ncols, null)        // raw data table
    val data    = preProcess (rawData)                             // cleaned-up data table
    val fname   = xcols.map (data.schema (_))                      // predictor feature/variable names
    val ofname  = "one" +: fname                                   // names with intercept/one (1) term added
    val rname   = data.schema (ycol)                               // name of the response column
    val (x, y)  = data.toMatrixV (xcols, ycol)                     // (predictor matrix, response column)
    val xy      = x +^ y                                           // combined xy matrix
    val ox      = VectorD.one (x.dim) +^: x                        // predictor matrix with one (1) term added

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Pre-process the raw data that may contain missing values, strings, dates,
     *  id columns, zero-variance columns, and outliers. Return the cleaned-up data.
     *  @param tab  the table/linked-relation containing the raw data
     */
    def preProcess (tab: LTable): LTable =
        tab                                                        // FIX -- to be implemented
    end preProcess

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show/print the table/linked-relation containing the data.
     *  @param tab  the table/linked-relation containing the data
     */
    def show (tab: LTable = data): Unit = tab.show ()

end Dataset


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Theory` class provides a high-level unified way to run data science and
 *  machine learning models.
 */
case class Theory ():

    private val debug = debugf ("Theory", true)                     // debug function

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform Exploratory Data Analysis (EDA) using Simple Linear Regression y vs. x_j
     *  for each predictor variable x_j.
     *  FIX -- other more flexible options should be explored, exp. for time series
     *  @param dset  the dataset to be explored
     */
    def exploreData (dset: Dataset): Unit =
        println (dset.xy.corr)
        val (x, y, fname) = (dset.x, dset.y, dset.fname)
        for j <- x.indices2 do
            banner (s"Plot response y vs. predictor variable ${fname(j)}")
            val mod = SimpleRegression (x(?, j), y, Array ("one", fname(j)))
            mod.trainNtest ()()                                     // train and test the model
        end for
    end exploreData

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run all models in the given list.
     */
    def runModels (models: List [Model & Fit]): Unit =
        for mod <- models do
            banner (s"In-Sample Testing of model ${mod.modelName}")
            mod.inSample_Test ()                                    // In-Sample Testing of full dataset
            println (mod.summary ())                                // FIX - only shows for last horizon

            banner (s"Validation Testing of model ${mod.modelName}")
            mod.validate ()()                                       // TnT: Train on training-set, Test on testing-set
            println (mod.equation)                                  // print the prediction equation using parameters from validate
                                                                    // FIX -- override equation for each model

            banner (s"Cross-Validation Testing of model ${mod.modelName}")
            if mod.taskType == TaskType.Predict then
                val stats = mod.crossValidate ()                    // Multiple TnT with Cross Validation
                FitM.showQofStatTable (stats)
            else if mod.taskType == TaskType.Forecast then
                val fmod = mod.asInstanceOf [Forecaster & Fit]
                fmod.setSkip (0)                                    // can use values from training set to not skip any in test
                fmod.rollValidate ()                                // Multiple TnT with Rolling Validation
        end for
    end runModels

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Screen/reduce the features/variables based on model agnostic dependency/correlation
     *  analysis of the predictor and response variables.  Performs a quick pre-screening.
     *  @param mod  the model to simplify by using fewer predictor variables
     */
    def screenFeatures (mod: Model & Fit): Model & Fit =
        val xy = mod.getXy
        val (xcols, idx) = mod.screen (xy)()                        // x-columns and their indices
        debug ("screenModel", s"for ${mod.modelName} selected column idx = $idx")
        if mod.taskType == TaskType.Predict then
            val pmod = mod.asInstanceOf [Predictor & Fit]
            pmod.buildModel (xcols, newFname (mod.getFname, idx))
        else if mod.isInstanceOf [Forecaster_Reg & Fit] then
            val fmod = mod.asInstanceOf [Forecaster_Reg & Fit]
            println (s"fnames = ${stringOf (mod.getFname)}")
            fmod.convertReg2Forc (LSET.from (idx.toArray))
        else
            mod                                                     // model does not yet support screening
    end screenFeatures

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Select the features/variables based on model QoF metrics.  Requires the model
     *  to be run multiple times.
     *  @param mod   the model to simplify by using fewer predictor variables
     *  @param tech  the feature selection technique to use (defaults to Backward)
     */
    def selectFeatures (mod: Model & FeatureSelection & Fit,
                        tech: SelectionTech = SelectionTech.Backward): Model & Fit =
        banner (s"Feature Selection Technique: $tech")
        if mod.taskType == TaskType.Predict then
            given qk: Int = QoF.rSqBar.ordinal
            val rSq = mod.selectFeatures (tech, "many")._2          // R^2, R^2 bar, sMAPE, R^2 cv
            new PlotM (null, rSq, Regression.metrics, s"R^2 vs n for Regression with $tech", lines = true)
            mod.getBest.mod
        else if mod.isInstanceOf [Forecaster_Reg & Fit] then
            val fmod = mod.asInstanceOf [Forecaster_Reg & Fit]
            given qk: Int = QoF.smapeC.ordinal
            val rSq = mod.selectFeatures (tech, "none")._2          // R^2, R^2 bar, sMAPE, NO R^2 cv
            new PlotM (null, rSq, Regression.metrics, s"R^2 vs n for Regression with $tech", lines = true)
            fmod.convertReg2Forc (mod.getBest.mod_cols)
        else
            mod                                                     // model does not yet support feature selection
    end selectFeatures

end Theory


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `theoryTest` main method test the `Theory` class.
 *  > runMain scalation.theory.theoryTest
 */
@main def theoryTest (): Unit =

//  import scalation.mathstat.Transform.RootForm
    import scalation.modeling._
    import scalation.modeling.forecasting._

    val datasets = Map ("Covid19" -> Dataset ("Covid19", "covid_19_weekly.csv",
                                              17, Array (1, 3, 4, 5, 6, 7, 8, 9, 10), 2),
                        "Influenza" -> Dataset ("Influenza", "national_illness_clip.csv",
                                                8, Array (1, 2, 3, 4, 6, 7), 5))

    for (dname, dset) <- datasets do
        dset.show ()

        banner (s"Perform Exploratory Data Analysis (EDA) on dataset $dname")
        val theory = Theory ()
        theory.exploreData (dset)

        val models = List (new Regression (dset.ox, dset.y, dset.ofname),
                           new TranRegression (dset.ox, dset.y, dset.ofname, yℱ = RootForm ()),
                           SymbolicRegression.quadratic (dset.x, dset.y, dset.fname),
                           ARX (dset.x, dset.y, 6, dset.fname),
                           ARX_Quad (dset.x, dset.y, 6, dset.fname),
                           ARX_SR (dset.x, dset.y, 6, dset.fname))

        banner (s"Apply the following models to dataset $dname")
        for mod <- models do println (mod.modelName)

        banner (s"Analyze dataset $dname using full models")
        theory.runModels (models)

        banner (s"Analyze dataset $dname using feature screened models")
        val models2 = models.map (theory.screenFeatures (_))
        theory.runModels (models2)

        banner (s"Analyze dataset $dname using feature selected best models")
        val models3 = models.map (theory.selectFeatures (_))        // from full models
//      val models3 = models2.map (theory.selectFeatures (_))       // from screened models
        theory.runModels (models3)
/*
*/

end theoryTest

