//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Tue Oct 13 16:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Dataset Configurations -- the Response, the Pool of
 *           Exogenous Variables and Who Picks Them
 *
 *  A `DatasetConfig` names what an experiment runs ON: the response, the POOL of
 *  exogenous variables (keys of the named-column map its `load` builds from the
 *  `Example_*` loaders or straight from a CSV file), the domain sentence a named
 *  prompt shows, the sampling frequency word of the blind prompt, the coefficient
 *  signs domain knowledge expects, and who picks the exogenous variables actually
 *  used (`ExoMode`: `Given` takes the list as is, `Auto` lets `ExoScreen` pick
 *  k of the pool on the training rows, @see `shortlist`).  `Datasets` holds the
 *  configurations of the paper's experiments: Covid-19 (three pools and the
 *  file-based one), ILI, ETTh1 (all rows and the first year) and GEFCom2014.
 *  This file depends only on the base: the `Example_*` loaders, `ExoScreen` and
 *  `MatrixD.loadH`; a search built on the forecasting package (its scorer passes
 *  its data, training rows and settings to `shortlist`) depends on it, never the
 *  other way round.  rev4.0: moved out of `SearchArm.scala`; rev4.1: moved from
 *  `nonlinear/llm_narx_sr` to `forecasting`, `shortlist` takes the data, the
 *  training rows, k and maxLag instead of a scorer.
 *  > runMain scalation.modeling.forecasting.datasetsTest
 *  > runMain scalation.modeling.forecasting.datasetsTest2
 *  @note    AI Assisted Code
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ExoMode` enum says who picks the exogenous variables: `Given` uses the
 *  configuration's list as is, `Auto` lets `ExoScreen` pick from it (the pool).
 */
enum ExoMode:
    case Given, Auto
end ExoMode


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DatasetConfig` case class names a dataset's response and its POOL of
 *  exogenous variables (keys of the dataset map its `load` builds), the domain
 *  sentence used by a named prompt, the sampling frequency word used by the blind
 *  prompt, the coefficient signs domain knowledge expects, and who picks the
 *  exogenous variables actually used (@see `ExoMode`, `shortlist`).
 *  @param name           the dataset name for banners and titles (never sent to the LLM)
 *  @param resp           the response variable name (the key "y")
 *  @param exo            the exogenous variable names available (keys of the map)
 *  @param schema         the domain sentence (only a named prompt shows it)
 *  @param expectedSigns  the expected coefficient sign per variable (+1/-1; unknown => absent)
 *  @param freq           the sampling frequency word for the generic sentence
 *  @param load           the loader building the named-column map (full series)
 *  @param exoMode        `Given` => use `exo` as is; `Auto` => `ExoScreen` picks k of them (@see `shortlist`)
 */
case class DatasetConfig (name: String, resp: String, exo: Seq [String], schema: String,
                          expectedSigns: Map [String, Int], freq: String,
                          load: () => Map [String, VectorD], exoMode: ExoMode = ExoMode.Given):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the exogenous variables an experiment builds on: the list itself
     *  (`Given`) or the screening's shortlist (`Auto`): `ExoScreen.shortlist` of
     *  the pool on the first trSize rows of data, the k best by lead cross-correlation
     *  up to maxLag (a formula search built on this package passes its scorer's
     *  data, training rows and settings).
     *  @param data    the named-column map (the response under `resp`, the pool under their names)
     *  @param trSize  the number of training rows the screening looks at
     *  @param k       the number of exogenous variables kept
     *  @param maxLag  the largest lead examined
     */
    def shortlist (data: Map [String, VectorD], trSize: Int, k: Int, maxLag: Int): Seq [String] =
        exoMode match
        case ExoMode.Given => exo
        case ExoMode.Auto  =>
            ExoScreen.shortlist (data (resp), exo.map (v => v -> data (v)).toMap, maxLag, trSize, k)._2
    end shortlist

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the plain-ARX baseline formula string: full contiguous lag windows
     *  on the response and the given exogenous variables, no transforms.
     *  @param maxLag  the lag window size
     *  @param exo_    the exogenous variables (defaults to the pool)
     */
    def baselineFormulaStr (maxLag: Int, exo_ : Seq [String] = exo): String =
        (resp +: exo_).map (v => s"$v~($maxLag,1)").mkString (s"$resp ~= ", " + ", "")
    end baselineFormulaStr

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the blind display name per variable (y, x1, x2, ...) over the pool.
     */
    def blindNames: Map [String, String] =
        ((resp -> "y") +: exo.zipWithIndex.map ((v, i) => v -> s"x${i+1}")).toMap
    end blindNames

end DatasetConfig


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Datasets` object holds the dataset configurations of the experiments;
 *  each `load` builds the named-column map from the example loaders.
 */
object Datasets:

    val COVID2_SCHEMA =
        "Dataset: weekly US Covid-19; y = new deaths; icu (patients in intensive care) " +
        "and pRate (share of tests that are positive) are leading indicators of deaths."

    val COVID_SCHEMA =
        "Dataset: weekly US Covid-19; y = new deaths; hosp and icu (patients " +
        "in hospital / intensive care) are leading indicators of deaths."

    val COVID_ALL_SCHEMA =
        "Dataset: weekly US Covid-19; y = new deaths; the candidate drivers are new cases, the " +
        "reproduction rate, icu and hospital patients, tests, the positivity rate, tests per case, " +
        "new vaccinations and excess mortality."

    val ILI_SCHEMA =
        "Dataset: weekly US influenza-like illness; y = total ILI patient count; " +
        "wili = population-weighted ILI percentage; ot = total patient visits."

    /** Covid-19 weekly deaths with icu patients and test positivity rate (`Example_Covid.BEST2_EXO`): we choose.
     */
    val COVID2 = DatasetConfig ("Covid-19 (icu, pRate)", "y", Seq ("icu", "pRate"), COVID2_SCHEMA,
                                Map ("icu" -> 1, "pRate" -> 1), "weekly",
                                () => Example_Covid.context (Example_Covid.BEST2_EXO, Array ("icu", "pRate")))

    /** Covid-19 weekly deaths with hospital and icu patients.
     */
    val COVID  = DatasetConfig ("Covid-19 (hosp, icu)", "y", Seq ("hosp", "icu"), COVID_SCHEMA,
                                Map ("hosp" -> 1, "icu" -> 1), "weekly",
                                () => Example_Covid.context (Array ("hosp_patients", "icu_patients"), Array ("hosp", "icu")))

    /** The pool of Covid-19 columns an automatic selection may pick from (the
     *  non-cumulative columns of `Example_Covid.header`) and their short names.
     */
    val COVID_POOL  = Array ("new_cases", "reproduction_rate", "icu_patients", "hosp_patients", "new_tests",
                             "positive_rate", "tests_per_case", "new_vaccinations", "excess_mortality")
    val COVID_NAMES = Array ("cases", "rRate", "icu", "hosp", "tests", "pRate", "tpc", "vacc", "exMort")

    /** Covid-19 weekly deaths with the whole pool: the system chooses (`ExoMode.Auto`).
     */
    val COVID_ALL = DatasetConfig ("Covid-19 (auto exo)", "y", COVID_NAMES.toSeq, COVID_ALL_SCHEMA,
                                   Map ("cases" -> 1, "rRate" -> 1, "icu" -> 1, "hosp" -> 1, "pRate" -> 1,
                                        "tpc" -> -1, "vacc" -> -1, "exMort" -> 1), "weekly",
                                   () => Example_Covid.context (COVID_POOL, COVID_NAMES), ExoMode.Auto)

    /** Influenza-like illness weekly counts with the weighted ILI percentage and total visits.
     */
    val ILI    = DatasetConfig ("ILI", "y", Seq ("wili", "ot"), ILI_SCHEMA,
                                Map ("wili" -> 1), "weekly",
                                () => Example_ILI.context (Array ("%WEIGHTED ILI", "OT"), Array ("wili", "ot")))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a configuration straight from a CSV file with a header line: the pool
     *  is every column `ExoScreen.pool` finds eligible (not the response, not
     *  excluded, finite, neither constant nor cumulative on the training rows),
     *  under its own column name, and the system chooses among them (`ExoMode.Auto`).
     *  No expected signs are known for such a pool.
     *  @param name      the dataset name for banners and titles
     *  @param fileName  the CSV file (relative to DATA_DIR), column names in its first line
     *  @param resp      the response column name
     *  @param schema    the domain sentence for a named prompt
     *  @param freq      the sampling frequency word
     *  @param rng       the rows to keep (null => all rows)
     *  @param exclude   the columns domain knowledge rules out (proxies of the response)
     *  @param skipCol   the leading columns to skip (e.g., a date column)
     */
    def fromFile (name: String, fileName: String, resp: String, schema: String, freq: String,
                  rng: Range = null, exclude: Set [String] = Set (), skipCol: Int = 1): DatasetConfig =
        val (all, header) = MatrixD.loadH (fileName, skipCol)
        val data = if rng == null then all else all(rng)
        val p    = ExoScreen.pool (data, header, resp, Model.trSize (data.dim), exclude)
        for (nm, why) <- p.dropped do println (s"Datasets.fromFile: $nm dropped -- $why")
        val map  = Map ("y" -> p.y) ++ p.xe
        DatasetConfig (name, "y", p.xe.keys.toSeq, schema, Map (), freq, () => map, ExoMode.Auto)
    end fromFile

    val COVID_FILE_SCHEMA =
        "Dataset: weekly US Covid-19; y = new deaths; the other columns of the file are the candidate drivers."

    /** Covid-19 weekly deaths straight from the file (the rows of `Example_Covid.clip`):
     *  the pool is whatever the screening finds eligible, minus excess mortality
     *  (another count of the same deaths) and new cases (not comparable across testing
     *  regimes) -- the two columns `Example_Covid` never uses; the system chooses.
     *  Lazy: the file is read when first used.
     */
    lazy val COVID_FILE = fromFile ("Covid-19 (file, auto exo)", Example_Covid.fileName, Example_Covid.response,
                                    COVID_FILE_SCHEMA, "weekly", 0 until 116, Set ("excess_mortality", "new_cases"))

    val ETTH1_SCHEMA =
        "Dataset: hourly electricity transformer (ETTh1); y = oil temperature of the transformer; " +
        "the candidate drivers are the high, middle and low useful/useless power loads " +
        "(hufl, hull, mufl, mull, lufl, lull); the oil warms with load and cools slowly."

    val GEF14_SCHEMA =
        "Dataset: daily electric load of a US utility (GEFCom2014); y = daily peak load; " +
        "temp and tmax are the daily mean and maximum temperature (deg F); load is lowest at " +
        "mild temperatures and rises in cold (heating) and hot (cooling) weather."

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the ETTh1 configuration: hourly oil temperature with the six loads as
     *  the pool, the system choosing k of them (`ExoMode.Auto`); every load
     *  is expected to warm the oil (+1).
     *  @param rng  the rows to keep (null => all 17420 rows)
     */
    def ettH1 (rng: Range = null): DatasetConfig =
        val nm = if rng == null then "ETTh1 (auto exo)" else s"ETTh1 rows ${rng.start}..${rng.end-1} (auto exo)"
        DatasetConfig (nm, "y", Example_ETTh1.LOAD_NAMES.toSeq, ETTH1_SCHEMA,
                       Example_ETTh1.LOAD_NAMES.map (_ -> 1).toMap, "hourly",
                       () => Example_ETTh1.context (rng = rng), ExoMode.Auto)
    end ettH1

    /** ETTh1, the full series (17420 hourly rows) and its first year (8760 rows,
     *  2016-07-01 .. 2017-06-30: one cycle of the seasons at a tenth of the cost).
     */
    val ETTH1    = ettH1 ()
    val ETTH1_1Y = ettH1 (0 until 8760)

    /** GEFCom2014 daily peak load with the daily mean and maximum temperature: we
     *  choose (the relation is V-shaped, so no sign is expected).
     */
    val GEF14 = DatasetConfig ("GEFCom2014 load (temp, tmax)", "y", Seq ("temp", "tmax"), GEF14_SCHEMA,
                               Map (), "daily", () => Example_GEFCom.context (Array ("temp", "tmax")))

end Datasets


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `datasetsTest` main function builds the file-based Covid-19 configuration:
 *  the pool comes from the file's header through `ExoScreen.pool` (the five running
 *  totals dropped, excess mortality and new cases excluded), and the configuration's
 *  shortlist (`ExoMode.Auto`, k = 2, maxLag = 6: the defaults of a formula search)
 *  must be what `exoScreenTest2` finds: hospital patients and the positivity rate.
 *  The baseline formula string and the blind names are built on that pool.
 *  > runMain scalation.modeling.forecasting.datasetsTest
 */
@main def datasetsTest (): Unit =

    val (k, maxLag) = (2, 6)                                             // a formula search's exoK and maxLag
    val cfg = Datasets.COVID_FILE
    val dataset = cfg.load ()

    banner (s"${cfg.name}: pool of ${cfg.exo.size} from the file's header")
    println (cfg.exo.mkString (", "))
    val short = cfg.shortlist (dataset, Model.trSize (dataset("y").dim), k, maxLag)
    println (s"shortlist (k = $k): ${short.mkString (", ")}")
    println (s"baseline: ${cfg.baselineFormulaStr (maxLag, short)}")
    println (s"blind names: ${cfg.blindNames}")

    assert (cfg.exoMode == ExoMode.Auto, "a file-based configuration lets the system choose")
    assert (cfg.exo.size == 8 && ! cfg.exo.contains ("excess_mortality") && ! cfg.exo.contains ("new_cases") &&
            ! cfg.exo.contains ("people_vaccinated"), "the pool is the eligible columns minus the exclusions")
    assert (dataset.size == cfg.exo.size + 1 && dataset("y").dim == 116, "the map holds y and the pool, 116 rows")
    assert (short == Seq ("hosp_patients", "positive_rate"), "the shortlist of exoScreenTest2")
    assert (cfg.blindNames.size == 9 && cfg.blindNames ("y") == "y", "one blind name per variable")
    println ("datasetsTest: passed")

end datasetsTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `datasetsTest2` main function loads the ETTh1 (first year) and GEFCom2014
 *  configurations, prints the screening's shortlist of the six loads for ETTh1,
 *  the baseline formula strings and the blind names, and checks the shapes; a
 *  configuration whose data file is absent is reported and skipped (k = 2 and
 *  maxLag = 6, a formula search's defaults).
 *  > runMain scalation.modeling.forecasting.datasetsTest2
 */
@main def datasetsTest2 (): Unit =

    val (k, maxLag) = (2, 6)                                             // a formula search's exoK and maxLag
    for cfg <- Seq (Datasets.ETTH1_1Y, Datasets.GEF14) do
        val loaded = try Some (cfg.load ())
                     catch case e: Exception =>
                         println (s"${cfg.name}: data file not available -- skipping: ${e.getMessage}")
                         None
        for dataset <- loaded do
            val m  = dataset("y").dim
            val tr = Model.trSize (m)
            banner (s"${cfg.name}: $m rows, pool ${cfg.exo.mkString (", ")}, mode ${cfg.exoMode}")
            val short = cfg.shortlist (dataset, tr, k, maxLag)
            println (s"shortlist: ${short.mkString (", ")}")
            println (s"baseline: ${cfg.baselineFormulaStr (maxLag, short)}")
            println (s"blind names: ${cfg.blindNames}")
            println (s"training rows ${0 until tr}, testing rows ${Forecaster.teRng (m)}")
            assert (dataset.size == cfg.exo.size + 1, "the map holds y and the pool")
            assert (short.nonEmpty && short.size <= cfg.exo.size, "a non-empty shortlist within the pool")
            if cfg eq Datasets.ETTH1_1Y then
                assert (m == 8760 && short.size == k, "ETTh1: one year, k loads chosen")
            else
                assert (m == 2099 && short == Seq ("temp", "tmax"), "GEFCom2014: 2099 days, the given temperatures")
    end for
    println ("datasetsTest2: passed")

end datasetsTest2
