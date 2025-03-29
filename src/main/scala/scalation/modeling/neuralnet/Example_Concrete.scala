
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Mar 16 15:13:38 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Dataset: Concrete Production
 *
 *  @see archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test
 *  @see pdfs.semanticscholar.org/2aba/047471e3e1eef6fa0319b4a63d3dfceafb0b.pdf
 */

package scalation
package modeling
package neuralnet

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ExampleConcrete` class stores a medium-sized example dataset from the UCI Machine
 *  Learning Repository,
 *  "Abstract:  Concrete is a highly complex material.  The slump flow of concrete is not only
 *   determined by the water content, but that is also influenced by other concrete ingredients."
 */
object Example_Concrete:

    // Input Variables (7) (component kg in one M^3 concrete):
    // 1. Cement
    // 2. Blast Furnace Slag
    // 3. Fly Ash
    // 4. Water
    // 5. Super Plasticizer (SP)
    // 6. Coarse Aggregate
    // 7. Fine Aggregate
    // Output Variables (3):
    // 1. SLUMP (cm)
    // 2. FLOW (cm)
    // 3. 28-day Compressive STRENGTH (Mpa) 

    val x_fname  = Array ("Cement", "Furnace_Slag", "Fly_Ash", "Water", "Super_Plasticizer",
                          "Coarse_Aggregate", "Fine_Aggregate")

    val ox_fname = Array ("One", "Cement", "Furnace_Slag", "Fly_Ash", "Water", "Super_Plasticizer",
                          "Coarse_Aggregate", "Fine_Aggregate")

    // Index, Cement,   Slag, Fly ash, Water,   SP,   Coarse,   Fine,  SLUMP,  FLOW,  STRENGTH
    val xy = MatrixD ((103, 11),
        1,	273,	82,	105,	210,	9,	904,	680,	23,	62,	34.99,
        2,	163,	149,	191,	180,	12,	843,	746,	0,	20,	41.14,
        3,	162,	148,	191,	179,	16,	840,	743,	1,	20,	41.81,
        4,	162,	148,	190,	179,	19,	838,	741,	3,	21.5,	42.08,
        5,	154,	112,	144,	220,	10,	923,	658,	20,	64,	26.82,
        6,	147,	89,	115,	202,	9,	860,	829,	23,	55,	25.21,
        7,	152,	139,	178,	168,	18,	944,	695,	0,	20,	38.86,
        8,	145,	0,	227,	240,	6,	750,	853,	14.5,	58.5,	36.59,
        9,	152,	0,	237,	204,	6,	785,	892,	15.5,	51,	32.71,
        10,	304,	0,	140,	214,	6,	895,	722,	19,	51,	38.46,
        11,	145,	106,	136,	208,	10,	751,	883,	24.5,	61,	26.02,
        12,	148,	109,	139,	193,	7,	768,	902,	23.75,	58,	28.03,
        13,	142,	130,	167,	215,	6,	735,	836,	25.5,	67,	31.37,
        14,	354,	0,	0,	234,	6,	959,	691,	17,	54,	33.91,
        15,	374,	0,	0,	190,	7,	1013,	730,	14.5,	42.5,	32.44,
        16,	159,	116,	149,	175,	15,	953,	720,	23.5,	54.5,	34.05,
        17,	153,	0,	239,	200,	6,	1002,	684,	12,	35,	28.29,
        18,	295,	106,	136,	206,	11,	750,	766,	25,	68.5,	41.01,
        19,	310,	0,	143,	168,	10,	914,	804,	20.5,	48.2,	49.3,
        20,	296,	97,	0,	219,	9,	932,	685,	15,	48.5,	29.23,
        21,	305,	100,	0,	196,	10,	959,	705,	20,	49,	29.77,
        22,	310,	0,	143,	218,	10,	787,	804,	13,	46,	36.19,
        23,	148,	180,	0,	183,	11,	972,	757,	0,	20,	18.52,
        24,	146,	178,	0,	192,	11,	961,	749,	18,	46,	17.19,
        25,	142,	130,	167,	174,	11,	883,	785,	0,	20,	36.72,
        26,	140,	128,	164,	183,	12,	871,	775,	23.75,	53,	33.38,
        27,	308,	111,	142,	217,	10,	783,	686,	25,	70,	42.08,
        28,	295,	106,	136,	208,	6,	871,	650,	26.5,	70,	39.4,
        29,	298,	107,	137,	201,	6,	878,	655,	16,	26,	41.27,
        30,	314,	0,	161,	207,	6,	851,	757,	21.5,	64,	41.14,
        31,	321,	0,	164,	190,	5,	870,	774,	24,	60,	45.82,
        32,	349,	0,	178,	230,	6,	785,	721,	20,	68.5,	43.95,
        33,	366,	0,	187,	191,	7,	824,	757,	24.75,	62.7,	52.65,
        34,	274,	89,	115,	202,	9,	759,	827,	26.5,	68,	35.52,
        35,	137,	167,	214,	226,	6,	708,	757,	27.5,	70,	34.45,
        36,	275,	99,	127,	184,	13,	810,	790,	25.75,	64.5,	43.54,
        37,	252,	76,	97,	194,	8,	835,	821,	23,	54,	33.11,
        38,	165,	150,	0,	182,	12,	1023,	729,	14.5,	20,	18.26,
        39,	158,	0,	246,	174,	7,	1035,	706,	19,	43,	34.99,
        40,	156,	0,	243,	180,	11,	1022,	698,	21,	57,	33.78,
        41,	145,	177,	227,	209,	11,	752,	715,	2.5,	20,	35.66,
        42,	154,	141,	181,	234,	11,	797,	683,	23,	65,	33.51,
        43,	160,	146,	188,	203,	11,	829,	710,	13,	38,	33.51,
        44,	291,	105,	0,	205,	6,	859,	797,	24,	59,	27.62,
        45,	298,	107,	0,	186,	6,	879,	815,	3,	20,	30.97,
        46,	318,	126,	0,	210,	6,	861,	737,	17.5,	48,	31.77,
        47,	280,	92,	118,	207,	9,	883,	679,	25.5,	64,	37.39,
        48,	287,	94,	121,	188,	9,	904,	696,	25,	61,	43.01,
        49,	332,	0,	170,	160,	6,	900,	806,	0,	20,	58.53,
        50,	326,	0,	167,	174,	6,	884,	792,	21.5,	42,	52.65,
        51,	320,	0,	163,	188,	9,	866,	776,	23.5,	60,	45.69,
        52,	342,	136,	0,	225,	11,	770,	747,	21,	61,	32.04,
        53,	356,	142,	0,	193,	11,	801,	778,	8,	30,	36.46,
        54,	309,	0,	142,	218,	10,	912,	680,	24,	62,	38.59,
        55,	322,	0,	149,	186,	8,	951,	709,	20.5,	61.5,	45.42,
        56,	159,	193,	0,	208,	12,	821,	818,	23,	50,	19.19,
        57,	307,	110,	0,	189,	10,	904,	765,	22,	40,	31.5,
        58,	313,	124,	0,	205,	11,	846,	758,	22,	49,	29.63,
        59,	143,	131,	168,	217,	6,	891,	672,	25,	69,	26.42,
        60,	140,	128,	164,	237,	6,	869,	656,	24,	65,	29.5,
        61,	278,	0,	117,	205,	9,	875,	799,	19,	48,	32.71,
        62,	288,	0,	121,	177,	7,	908,	829,	22.5,	48.5,	39.93,
        63,	299,	107,	0,	210,	10,	881,	745,	25,	63,	28.29,
        64,	291,	104,	0,	231,	9,	857,	725,	23,	69,	30.43,
        65,	265,	86,	111,	195,	6,	833,	790,	27,	60,	37.39,
        66,	159,	0,	248,	175,	12,	1041,	683,	21,	51,	35.39,
        67,	160,	0,	250,	168,	12,	1049,	688,	18,	48,	37.66,
        68,	166,	0,	260,	183,	13,	859,	827,	21,	54,	40.34,
        69,	320,	127,	164,	211,	6,	721,	723,	2,	20,	46.36,
        70,	336,	134,	0,	222,	6,	756,	787,	26,	64,	31.9,
        71,	276,	90,	116,	180,	9,	870,	768,	0,	20,	44.08,
        72,	313,	112,	0,	220,	10,	794,	789,	23,	58,	28.16,
        73,	322,	116,	0,	196,	10,	818,	813,	25.5,	67,	29.77,
        74,	294,	106,	136,	207,	6,	747,	778,	24,	47,	41.27,
        75,	146,	106,	137,	209,	6,	875,	765,	24,	67,	27.89,
        76,	149,	109,	139,	193,	6,	892,	780,	23.5,	58.5,	28.7,
        77,	159,	0,	187,	176,	11,	990,	789,	12,	39,	32.57,
        78,	261,	78,	100,	201,	9,	864,	761,	23,	63.5,	34.18,
        79,	140,	1.4,	198.1,	174.9,	4.4,	1049.9,	780.5,	16.25,	31,	30.83,
        80,	141.1,	0.6,	209.5,	188.8,	4.6,	996.1,	789.2,	23.5,	53,	30.43,
        81,	140.1,	4.2,	215.9,	193.9,	4.7,	1049.5,	710.1,	24.5,	57,	26.42,
        82,	140.1,	11.8,	226.1,	207.8,	4.9,	1020.9,	683.8,	21,	64,	26.28,
        83,	160.2,	0.3,	240,	233.5,	9.2,	781,	841.1,	24,	75,	36.19,
        84,	140.2,	30.5,	239,	169.4,	5.3,	1028.4,	742.7,	21.25,	46,	36.32,
        85,	140.2,	44.8,	234.9,	171.3,	5.5,	1047.6,	704,	23.5,	52.5,	33.78,
        86,	140.5,	61.1,	238.9,	182.5,	5.7,	1017.7,	681.4,	24.5,	60,	30.97,
        87,	143.3,	91.8,	239.8,	200.8,	6.2,	964.8,	647.1,	25,	55,	27.09,
        88,	194.3,	0.3,	240,	234.2,	8.9,	780.6,	811.3,	26.5,	78,	38.46,
        89,	150.4,	110.9,	239.7,	168.1,	6.5,	1000.2,	667.2,	9.5,	27.5,	37.92,
        90,	150.3,	111.4,	238.8,	167.3,	6.5,	999.5,	670.5,	14.5,	36.5,	38.19,
        91,	155.4,	122.1,	240,	179.9,	6.7,	966.8,	652.5,	14.5,	41.5,	35.52,
        92,	165.3,	143.2,	238.3,	200.4,	7.1,	883.2,	652.6,	17,	27,	32.84,
        93,	303.8,	0.2,	239.8,	236.4,	8.3,	780.1,	715.3,	25,	78,	44.48,
        94,	172,	162.1,	238.5,	166,	7.4,	953.3,	641.4,	0,	20,	41.54,
        95,	172.8,	158.3,	239.5,	166.4,	7.4,	952.6,	644.1,	0,	20,	41.81,
        96,	184.3,	153.4,	239.2,	179,	7.5,	920.2,	640.9,	0,	20,	41.01,
        97,	215.6,	112.9,	239,	198.7,	7.4,	884,	649.1,	27.5,	64,	39.13,
        98,	295.3,	0,	239.9,	236.2,	8.3,	780.3,	722.9,	25,	77,	44.08,
        99,	248.3,	101,	239.1,	168.9,	7.7,	954.2,	640.6,	0,	20,	49.97,
        100,	248,	101,	239.9,	169.1,	7.7,	949.9,	644.1,	2,	20,	50.23,
        101,	258.8,	88,	239.6,	175.3,	7.6,	938.9,	646,	0,	20,	50.5,
        102,	297.1,	40.9,	239.9,	194,	7.5,	908.9,	651.8,	27.5,	67,	49.17,
        103,	348.7,	0.1,	223.1,	208.5,	9.6,	786.2,	758.1,	29,	78,	48.77)

    val (min_x, max_x) = (xy.min, xy.max)

    val xy_s  = scale ((min_x, max_x), (0, 1)) (xy)              // column-wise scaled to [0.0, 1.0]
    val xy_s2 = scale ((min_x, max_x), (-1, 1)) (xy)             // column-wise scaled to [-1.0, 1.0]
    xy_s.setCol (0, 1)                                           // turn index column into a column of all ones
    xy_s2.setCol (0, 1)                                          // turn index column into a column of all ones

    val ox = xy_s(?, 0 until 8)                                  // input matrix - include column 0 for intercept
    val x  = xy_s(?, 1 until 8)                                  // input matrix - exclude column 0 for no intercept
    val y  = xy_s(?, 8 until 11)                                 // output matrix

end Example_Concrete

import Example_Concrete._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_ConcreteTest` main function tests the `Example_Concrete` object.
 *  These test cases compare several modeling techniques.  This one runs `Regression`
 *  and `RegressionMV`.
 *  > runMain scalation.modeling.neuralnet.example_ConcreteTest
 */
@main def example_ConcreteTest (): Unit =

    println (s"input ox: ${ox.dim}, ${ox.dim2}")
    println (s"output y: ${y.dim}, ${y.dim2}")

    for j <- y.indices2 do
        banner (s"Concrete - Regression for y$j")
        val yj  = y(?, j)                                        // use j-th column of matrix y
        val mod = new Regression (ox, yj, ox_fname)              // create j-th Regression model
        mod.trainNtest ()()                                      // train and test the model
        println (mod.summary ())                                 // parameter/coefficient statistics
    end for

    banner ("Concrete - RegressionMV")
    val mod = new RegressionMV (ox, y, ox_fname)                 // create model with intercept (else pass x)
    val (yp, qof) = mod.trainNtest ()()                          // train and test the model
    println (mod.summary ())                                     // parameter/coefficient statistics

    val sse = qof (QoF.sse.ordinal)
    val sst = qof (QoF.sst.ordinal)
    val sse_all = sse.sum
    val sst_all = sst.sum

    println ("sst_all = " + sst_all)
    println ("sse_all = " + sse_all)
    println ("rSq_all = " + (sst_all - sse_all) / sst_all)

end example_ConcreteTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_ConcreteTest2` main function is used to test the `ExampleConcrete` object.
 *  These test cases compare several modeling techniques.  This one runs `Perceptron`.
 *  with trainNtest (manual hyper-parameter tuning).
 *  > runMain scalation.modeling.neuralnet.example_ConcreteTest2
 */
@main def example_ConcreteTest2 (): Unit =

    banner ("Perceptron")

    println (s"input ox: ${ox.dim}, ${ox.dim2}")
    println (s"output y: ${y.dim}, ${y.dim2}")

    var sst_all = 0.0
    var sse_all = 0.0

    for j <- y.indices2 do
        val yj = y(?, j)
        Perceptron.hp("eta") = 0.1                               // try several values for the learning rate eta
        val pt = Perceptron.rescale (ox, yj)

        banner (s"Perceptron: trainNtest for $j th column")
        pt.trainNtest ()()                                       // try train vs. train0
        val yp  = pt.predict (x)                                 // predicted output values
        val e   = yj - yp
        val sse = e dot e
        val sst = (yj dot yj) - yj.sum~^2 / yj.dim
        println ("sse = " + sse)
        sse_all += sse
        sst_all += sst

        banner ("predicted output")
//      println ("diff: y - yp = " + (y - yp))
        new Plot (null, yj, yp, s"y vs yp for $j th column")
    end for

    println ("sst_all = " + sst_all)
    println ("sse_all = " + sse_all)
    println ("rSq_all = " + (sst_all - sse_all) / sst_all)

end example_ConcreteTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_ConcreteTest3` main function is used to test the `ExampleConcrete` object.
 *  These test cases compare several modeling techniques.  This one runs `NeuralNet_2L`
 *  (that effectively is multiplke perceptrons) with trainNtest2 (partially automated
 *  hyper-parameter tuning).  Compare the overall R^2 for these three test cases.
 *  > runMain scalation.modeling.neuralnet.example_ConcreteTest3
 */
@main def example_ConcreteTest3 (): Unit =

    banner ("Perceptron with train2")

    println (s"input ox: ${ox.dim}, ${ox.dim2}")
    println (s"output y: ${y.dim}, ${y.dim2}")

    var sst_all = 0.0
    var sse_all = 0.0

    val pt = NeuralNet_2L.rescale (ox, y)

    banner (s"Neural_2L as 3 perceptrons: trainNtest2")
    val (yp, qof) = pt.trainNtest2 ()()                          // interval search on eta
    for j <- y.indices2 do
        val yj  = y(?, j)
        val ypj = yp(?, j)
        val e   = yj - ypj
        val sse = e dot e
        val sst = (yj dot yj) - yj.sum~^2 / yj.dim
        sse_all += sse
        sst_all += sst
        new Plot (null, yj, ypj, s"y vs yp for $j th column")
    end for

    println ("sst_all = " + sst_all)
    println ("sse_all = " + sse_all)
    println ("rSq_all = " + (sst_all - sse_all) / sst_all)

end example_ConcreteTest3

