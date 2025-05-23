
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Santosh Uttam Bobade, John Miller
 *  @version 2.0
 *  @date    Sat Aug 11 16:00:10 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note   Class for Latitude-Longitude coordinates
 *
 *  Support Latitude-Longitude coordinates as well as conversions from
 *  LatLong coordinates to UTM coordinates and UTM coordinates to
 *  LatLong coordinates.
 *
 *  LatLong coordinates:
 *  @see https://en.wikipedia.org/wiki/Geographic_coordinate_system
 *
 *  UTM coordinates:
 *  @see en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system
 *
 *  Adapted from CoordinateConversion.java
 *  Author: Sami Salkosuo, sami.salkosuo@fi.ibm.com
 *  (c) Copyright IBM Corp. 2007
 *  @see www.ibm.com/developerworks/library/j-coordconvert/index.html
 */

package scalation

import scala.math._
import scala.util.control.Breaks.{breakable, break}

import Earth._

val CENTRAL_MERIDIAN = 500000        // 500,000 meters designated for Central Meridian for each of the 60 Zones
val EQUATOR          = 10000000      // 10,000,000 meters designated for Equator for Southern Hemisphere, 0 for Northern

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LatLong` class stores Latitude-Longitude coordinates.
 *  Note: when converting from String to a number use 'mk' not 'to' (e.g., mkInt not toInt).
 *  @param lat   the latitude in degrees North of the Equator
 *  @param long  the longitude in degrees East of the Prime Meridian
 */
case class LatLong (lat: Double, long: Double):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Construct a `LatLong` object from a pair of doubles.
     *  @param latLong  the pair of doubles
     */
    def this (latLong: (Double, Double)) = this (latLong._1, latLong._2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Construct a `LatLong` object from latitude and longitude as DMS strings.
     *  @param latDMS   the latitude in Degrees, Minutes, Seconds 
     *  @param longDMS  the longitude in Degrees, Minutes, Seconds
     */
    def this (latDMS: (Int, Int, Int), longDMS: (Int, Int, Int)) =
        this (latDMS._1  + latDMS._2 / 60.0  + latDMS._3 / 3600.0,
              longDMS._1 + longDMS._2 / 60.0 + longDMS._3 / 3600.0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the latitude and longitude are invalid.
     */
    def invalid: Boolean = lat < -90.0 || lat > 90.0 || long < -180.0 || long >= 180.0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the distance between this Latitude-Longitude and location loc2 in meters.
     *  @param loc2       the other Latitude-Longitude location
     *  @param elevation  the average elevation for two locations (default 0 => sea level)
     */
    def distance (loc2: LatLong, elevation: Double = 0.0): Double =
        val latDiff  = toRadians (lat - loc2.lat)
        val longDiff = toRadians (long - loc2.long)
        val sinLat   = sin (latDiff / 2)
        val sinLong  = sin (longDiff / 2)
        val a = sinLat * sinLat + (cos (toRadians (lat)) * cos (toRadians (loc2.lat)) * sinLong * sinLong)
        val d = 2 * atan2 (sqrt (a), sqrt (1 - a))                  // distance for unit radius
        d * (meanRadius + elevation)                                // distance for Earth radius
    end distance

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the kappa nearest neighbors (top-kappa) to the this location, returning
     *  the indices and distances where they are found in locs.
     *  @param locs   the other locations to be considered
     *  @param kappa  the number of nearest neighbors to collect
     */
    def kNearest (locs: Array [LatLong], kappa: Int = 3): Array [(Int, Double)] =
        val k    = min (locs.length, kappa)
        val topK = Array.fill [(Int, Double)] (k)(-1, Double.PositiveInfinity)  // top-k nearest points (in reserve order)
        var dk   = Double.PositiveInfinity
        for i <- locs.indices do
            val di = distance (locs(i))                             // compute distance this to locs(i)
            if di < dk then dk = replaceTop (topK, i, di)           // if closer, adjust top-kappa
        end for
        topK
    end kNearest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove the most distant neighbor and add new neighbor i.  Maintain the
     *  topK nearest neighbors in sorted order farthest to nearest.
     *  @param topK  the top-k nearest points (in reserve order)
     *  @param i     new neighbor to be added
     *  @param di    distance of the new neighbor
     */
    private def replaceTop (topK: Array [(Int, Double)], i: Int, di: Double): Double =
        val k = topK.length
        var j = 0
        while j < k-1 && di < topK(j)._2 do { topK(j) = topK(j+1); j += 1 }
        topK(j) = (i, di)
        topK(0)._2                                                  // distance of new farthest neighbor
    end replaceTop

end LatLong


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Earth` object stores basic Earth properties.
 *  @see en.wikipedia.org/wiki/Earth_radius
 */
object Earth:

    val equatorialRadius = 6378137.0                                // equatorial radius
    val polarRadius      = 6356752.314                              // polar radius
    val meanRadius       = 6371008.8                                // IUGG mean radius sea level
    val meanRadiusEl     = 6371230.0                                // average radius elevation
//  val k0   = polarRadius / equatorialRadius                       // ratio of radii
    val k0   = 0.9996
//  val e    = sqrt (1 - pow (polarRadius / equatorialRadius, 2))   // eccentricity
    val e    = 0.081819191
//  val e1sq = e * e / (1 - e * e)                                  // ratio
    val e1sq = 0.006739497

    val meters2Miles = 0.000621371                                  // meters to miles
    val miles2Meters = 1609.34                                      // meters to miles

    val a0   = 6367449.146
    val b0   = 16038.42955
    val c0   = 16.83261333
    val d0   = 0.021984404
    val e0   = 0.000312705
end Earth


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LatLong2UTM` object support conversion from Latitude-Longitude coordinates
 *  to Universal Transverse Mercator (UTM) coordinates.
 */
object LatLong2UTM:

    private val flaw = flawf ("LatLong2UTM")                        // flaw function

    private val negLetters = Array ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M')
    private val posLetters = Array ('N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Z')
    private val negDegrees = Array (-90, -84, -72, -64, -56, -48, -40, -32, -24, -16, -8)
    private val posDegrees = Array (  0,   8,  16,  24,  32,  40,  48,  56,  64,  72, 84)

    private var p = 0.0                                             // r curv 2
    private var k1, k2, k3, k4, k5 = 0.0                            // coefficients for UTM Coordinates

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert Latitude-Longitude to UTM coordinates as a string.
     *  @param ll  the Latitude-Longitude
     */
    def latLong2UTM (ll: LatLong): String =
        if ll.invalid then flaw ("latLong2UTM", s"invalid LatLong = $ll")
        setVariables (ll.lat, ll.long)
        val longZone = getLongZone (ll.long)
        val latZone  = getLatZone (ll.lat)
        longZone + " " + latZone + " " + getEasting.toInt + " " + getNorthing (ll.lat).toInt
    end latLong2UTM

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert Latitude-Longitude to (x, y) UTM coordinates (only valid within one zone).
     *  @param ll  the Latitude-Longitude
     */
    def latLong2UTMxy (ll: LatLong): (Double, Double) =
        if ll.invalid then flaw ("latLong2UTMxy", s"invalid LatLong = $ll")
        setVariables (ll.lat, ll.long)
        (getEasting.toInt, getNorthing (ll.lat).toInt)
    end latLong2UTMxy
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert Latitude-Longitude to UTM coordinate zone.
     *  @param ll  the Latitude-Longitude
     */
    def latLong2UTMzone (ll: LatLong): (String, String) =
        if ll.invalid then flaw ("latLong2UTMzone", s"invalid LatLong = $ll")
        setVariables (ll.lat, ll.long)
        (getLongZone (ll.long), getLatZone (ll.lat))
    end latLong2UTMzone

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set variable used in calculations
     *  @param lat   the latitude
     *  @param long  the longitude
     */
    private def setVariables (lat: Double, long: Double): Unit =
        val sin1 = 4.84814E-06
        val sin2 = sin1 * sin1
        val latR = toRadians (lat)
        val nu   = equatorialRadius / sqrt (1 -  pow (e * sin (latR), 2))
        val var1 = if long < 0.0 then ((180 + long) / 6.0).toInt + 1 else (long / 6).toInt + 31
        val var2 = 6 * var1 - 183
        val var3 = long - var2
        p        = var3 * 0.36
        val s    = a0 * latR - b0 * sin (2 * latR) + c0 * sin (4 * latR) - d0 * sin (6 * latR) + e0 * sin (8 * latR)
        k1       = s * k0
        k2       = nu * sin (latR) * cos (latR) * sin2 * k0 * 50000000
        k3       = (nu * sin2 * sin2 * sin (latR) * pow (cos (latR), 3) / 24) *
                   (5 - pow (tan (latR), 2) + 9 * e1sq * pow ( cos(latR), 2) + 4 * e1sq * e1sq * pow (cos (lat), 4)) *
                   k0 * 10000000000000000L
        k4       = nu * cos (latR) * sin1 * k0 * 10000
        k5       = (nu / 6) * pow (sin1 * cos (latR), 3) * (1 - pow (tan (latR), 2) + e1sq * pow (cos (latR), 2)) *
                   k0 * 1000000000000L
    end setVariables

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the longitude zone.
     *  @param long  the longitude
     */
    private def getLongZone (long: Double): String =
        val longZone = if long < 0.0 then (180.0 + long) / 6 + 1 else long / 6 + 31
        val value    = longZone.toInt.toString
        if value.length == 1 then "0" + value else value
    end getLongZone

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the latitude zone.
     *  @param lat  the latitude
     */
    private def getLatZone (lat: Double): String =
        var latIndex = -2
        val ilat     = lat.toInt

        if ilat >= 0 then
            breakable {
                for i <- posLetters.indices do
                    if ilat == posDegrees(i) then { latIndex = i; break () }
                    if ilat <  posDegrees(i) then { latIndex = i - 1; break () }
                end for
            } // breakable
        else
            breakable {
                for i <- negLetters.indices do
                    if ilat == negDegrees(i) then { latIndex = i; break () }
                    if ilat <  negDegrees(i) then { latIndex = i - 1; break () }
                end for
            } // breakable
        end if
        if latIndex == -1 then latIndex = 0
        if ilat >= 0 then
            if latIndex == -2 then latIndex = posLetters.length - 1
            posLetters (latIndex).toString
        else
            if latIndex == -2 then latIndex = negLetters.length - 1
            negLetters (latIndex).toString
        end if
    end getLatZone

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the number of meters East of zone reference.
     */
    private def getEasting: Double = CENTRAL_MERIDIAN + (k4 * p + k5 * pow (p, 3))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the number of meters North of zone reference.
     *  @param lat  the latitude
     */
    private def getNorthing (lat: Double): Double =
        val northing = k1 + k2 * p * p + k3 * pow (p, 4)
        if lat < 0.0 then EQUATOR + northing else northing
    end getNorthing

end LatLong2UTM


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The LatLong2CTM object supports conversion from latlong coordinates to CTM
 *  (Custom Transverse Mercator) coordinates
 */
object LatLong2CTM:

    private val flaw = flawf ("LatLong2CTM")                        // flaw function

    private var p = 0.0                                             // r curv 2
    private var k1, k2, k3, k4, k5 = 0.0                            // coefficients for UTM Coordinates
   
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert Latitude-Longitude to (x, y) CTM (Custom Transverse Mercator)
     *  coordinates.
     *  @param ll    the Latitude-Longitude
     *  @param cent  the central meridian of the custom TM zone
     */
    def latLong2CTMxy (ll: LatLong, cent: Double): (Double, Double) =
        if ll.invalid then flaw ("latLong2CTMxy", s"invalid LatLong = $ll")
        setVariables (ll.lat, ll.long, cent)
        (getEasting.toInt, getNorthing (ll.lat).toInt)
    end latLong2CTMxy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set variables used in calculations.
     *  @param lat   the latitude
     *  @param long  the longitude
     *  @param cent  the central meridian of the custom TM zone
     */
    private def setVariables (lat: Double, long: Double, cent: Double): Unit =
        val sin1 = 4.84814E-06
        val sin2 = sin1 * sin1
        val latR = toRadians (lat)
        val nu   = equatorialRadius / sqrt (1 -  pow (e * sin (latR), 2))
        val var1 = long - cent
        p        = var1 * 0.36
        val s    = a0 * latR - b0 * sin (2 * latR) + c0 * sin (4 * latR) - d0 * sin (6 * latR) + e0 * sin (8 * latR)
        k1       = s * k0
        k2       = nu * sin (latR) * cos (latR) * sin2 * k0 * 50000000
        k3       = (nu * sin2 * sin2 * sin (latR) * pow (cos (latR), 3) / 24) *
                   (5 - pow (tan (latR), 2) + 9 * e1sq * pow ( cos(latR), 2) + 4 * e1sq * e1sq * pow (cos (lat), 4)) *
                   k0 * 10000000000000000L
        k4       = nu * cos (latR) * sin1 * k0 * 10000
        k5       = (nu / 6) * pow (sin1 * cos (latR), 3) * (1 - pow (tan (latR), 2) + e1sq * pow (cos (latR), 2)) *
                   k0 * 1000000000000L
    end setVariables

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the number of meters East of zone reference.
     */
    private def getEasting: Double = CENTRAL_MERIDIAN + (k4 * p + k5 * pow (p, 3))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the number of meters North of zone reference.
     *  @param lat  the latitude
     */
    private def getNorthing (lat: Double): Double =
        val northing = k1 + k2 * p * p + k3 * pow (p, 4)
        if lat < 0.0 then EQUATOR + northing else northing
    end getNorthing

end LatLong2CTM


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `UTM2LatLong` object support conversion from Universal_Transverse_Mercator
 *  (UTM) coordinates to Latitude-Longitude coordinates.
 */
object UTM2LatLong:

    private val debug = debugf ("UTM2LatLong", true)                // debug function

    private val southernHemisphere = "ACDEFGHJKLM"                  // latitude zones in southern hemisphere
    private var phi1  = 0.0                                         // variables set in setVariable used elsewhere
    private var fact1 = 0.0
    private var fact2 = 0.0
    private var fact3 = 0.0
    private var fact4 = 0.0
    private var _a3   = 0.0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether the latitude zone is in the Northern "N" or Southern "S"
     *  hemisphere.
     *  @param latZone  the latitude zone
     */
    def getHemisphere (latZone: String): String =
        if southernHemisphere.indexOf (latZone) > -1 then "S" else "N"
    end getHemisphere

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert UTM coordinates to Latitude-Longitude coordinates.
     *  @param _utm  the UTM coordinates
     */
    def uTM2LatLong (_utm: String): (Double, Double) =
        debug ("uTM2LatLong", s"_utm = $_utm")
        val utm      = _utm.split (" ")
        val zone     = utm(0).mkInt                                 // use mkInt instead of toInt
        val latZone  = utm(1)
        val easting  = utm(2).mkDouble                              // use mkDouble instead of toDouble
        val northing = utm(3).mkDouble
        debug ("uTM2LatLong", s"zone = $zone, latZone = $latZone, easting = $easting, northing = $northing")

        val hemisphere = getHemisphere (latZone)
        setVariables (easting, if hemisphere == "S" then EQUATOR - northing else northing)
        val lat    = 180 * (phi1 - fact1 * (fact2 + fact3 + fact4)) / Pi
        val zoneCM = if zone > 0 then 6 * zone - 183.0 else 3.0
        val long   = zoneCM - _a3
        (if hemisphere == "S" then -lat else lat, long)
    end uTM2LatLong

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set variable used in calculations.
     *  @param easting   the number if meters East from the Central Meridian of zone
     *  @param northing  the number of meters North from the Equator
     */
    private def setVariables (easting: Double, northing: Double): Unit =
        val arc  = northing / k0
        val mu   = arc / (equatorialRadius * (1 - pow (e, 2) / 4.0 - 3 * pow (e, 4) / 64.0 - 5 * pow (e, 6) / 256.0))
        val ei   = (1 - pow (1 - e * e, 1 / 2.0)) / (1 + pow (1 - e * e, 1 / 2.0))
        val ca   = 3 * ei / 2 - 27 * pow (ei, 3) / 32.0
        val cb   = 21 * pow (ei, 2) / 16 - 55 * pow (ei, 4) / 32
        val cc   = 151 * pow (ei, 3) / 96
        val cd   = 1097 * pow (ei, 4) / 512
        phi1     = mu + ca * sin (2 * mu) + cb * sin (4 * mu) + cc * sin (6 * mu) + cd * sin (8 * mu)
        val n0   = equatorialRadius / pow (1 - pow  ( (e * sin (phi1)), 2), 1 / 2.0)
        val r0   = equatorialRadius * (1 - e * e) / pow (1 - pow ( (e * sin (phi1)), 2), 3 / 2.0)
        fact1    = n0 * tan (phi1) / r0
        val _a1  = CENTRAL_MERIDIAN - easting
        val dd0  = _a1 / (n0 * k0)
        fact2    = dd0 * dd0 / 2
        val t0   = pow (tan (phi1), 2)
        val Q0   = e1sq * pow (cos (phi1), 2)
        fact3    = (5 + 3 * t0 + 10 * Q0 - 4 * Q0 * Q0 - 9 * e1sq) * pow (dd0, 4) / 24
        fact4    = (61 + 90 * t0 + 298 * Q0 + 45 * t0 * t0 - 252 * e1sq - 3 * Q0 * Q0) * pow (dd0, 6) / 720
        val lof1 = _a1 / (n0 * k0)
        val lof2 = (1 + 2 * t0 + Q0) * pow (dd0, 3) / 6.0
        val lof3 = (5 - 2 * Q0 + 28 * t0 - 3 * pow (Q0, 2) + 8 * e1sq + 24 * pow (t0, 2)) * pow (dd0, 5) / 120
        val _a2  = (lof1 - lof2 + lof3) / cos (phi1)
        _a3      = _a2 * 180 / Pi
    end setVariables

end UTM2LatLong


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `latLongTest` main function tests UTM to LatLong conversions in both directions.
 *  > runMain scalation.latLongTest
 */
@main def latLongTest (): Unit =

    import UTM2LatLong.uTM2LatLong
    import LatLong2UTM.latLong2UTM

    val uTM = Array ("31 N 166021 0",       "30 N 808084 14385",   "34 G 683473 4942631", "25 L 404859 8588690",
                     "02 C 506346 1057742", "60 Z 500000 9997964", "01 A 500000 2035",    "31 Z 500000 9997964",
                     "08 Q 453580 2594272", "57 X 450793 8586116", "22 A 502639 75072")

    val latLong = Array ((  0.0000,    0.0000),  ( 0.1300,  -0.2324), (-45.6456,   23.3545), (-12.7650, -33.8765),
                         (-80.5434, -170.6540),  (90.0000, 177.0000), (-90.0000, -177.0000),  (90.0000,   3.0000),
                         ( 23.4578, -135.4545),  (77.3450, 156.9876), (-89.3454, -48.9306))

    banner ("Test: UTM to LatLong")                                 // test UTM to lat-long conversion
    for i <- uTM.indices do
        val uTM_i    = uTM(i)
        val llResult = uTM2LatLong (uTM_i)
        println (s"llResult = $llResult")
        val (lat, long) = (roundTo (llResult._1), roundTo (llResult._2))
        println (s"  ${i+1}. \t uTM2LatLong ($uTM_i) = ($lat, $long)")
        assert ((lat, long)  == latLong(i))
    end for

    banner ("Test: LatLong to UTM")                                 // test lat-long to UTM conversion
    for i <- latLong.indices do
        val ll_i      = latLong(i)
        val uTMResult = latLong2UTM (new LatLong (ll_i))
        println (s"  ${i+1}. \t latLong2UTM ($ll_i) = $uTMResult")
        assert (uTMResult == uTM(i))
    end for

end latLongTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `latLongTest2` main function tests distance calculations based on
 *  latitude longitude locations for typical elevation (Athens, GA to Atlanta, GA).
 *  > runMain scalation.latLongTest2
 */
@main def latLongTest2 (): Unit =

    val athens     = LatLong (33.9519, -83.3576)                    // Athens, GA
    val athens_El  = 214.0                                          // Athens elevation
    val atlanta    = LatLong (33.7490, -84.3880)                    // Atlanta, GA
    val atlanta_El = 320.0                                          // Atlanta elevation
    val el         = (athens_El + atlanta_El) / 2.0                 // average elevation

    var d = athens.distance (atlanta)
    banner ("Athens to Atlanta at elevation " + 0.0)
    println (s"Distance from Athens to Atlanta = $d meters.")
    println (s"Distance from Athens to Atlanta = ${d/1000} kilometers.")
    println (s"Distance from Athens to Atlanta = ${d*meters2Miles} miles.")

    d = athens.distance (atlanta, el)
    banner ("Athens to Atlanta at elevation " + el)
    println (s"Distance from Athens to Atlanta = $d meters.")
    println (s"Distance from Athens to Atlanta = ${d/1000} kilometers.")
    println (s"Distance from Athens to Atlanta = ${d*meters2Miles} miles.")

end latLongTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `latLongTest3` main function tests distance calculations based on
 *  latitude longitude locations for high elevation (Aspen, CO to Breckenridge, CO).
 *  > runMain scalation.latLongTest3
 */
@main def latLongTest3 (): Unit =

    val aspen      = LatLong (39.192297, -106.82447)                // Aspen, CO
    val aspen_El   = 2438.4                                         // Aspen elevation
    val brecken    = LatLong (39.486445, -106.043516)               // Breckenridge, CO
    val brecken_El = 2926.0                                         // Breckenridge elevation
    val el         = (aspen_El + brecken_El) / 2.0                  // average elevation

    // Note: DMS is less precise than decimal
    banner ("Aspen, CO at        " + new LatLong ((39, 11, 32), (106, 49, 28)))
    banner ("Breckenridge, CO at " + new LatLong ((39, 29, 11), (106,  2, 37)))

    var d = aspen.distance (brecken)
    banner ("Aspen to Breckenridge at elevation " + 0.0)
    println (s"Distance from Aspen to Breckenridge = $d meters.")
    println (s"Distance from Aspen to Breckenridge = ${d/1000} kilometers.")
    println (s"Distance from Aspen to Breckenridge = ${d*meters2Miles} miles.")

    d = aspen.distance (brecken, el)
    banner ("Aspen to Breckenridge  at elevation " + el)
    println (s"Distance from Aspen to Breckenridge = $d meters.")
    println (s"Distance from Aspen to Breckenridge = ${d/1000} kilometers.")
    println (s"Distance from Aspen to Breckenridge = ${d*meters2Miles} miles.")

end latLongTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `latLongTest4` main function tests the kNearest method used to find
 *  locations near by to Athens, GA.
 *  > runMain scalation.latLongTest4
 */
@main def latLongTest4 (): Unit =

    val sensorLoc = LatLong (33.9519, -83.3576)                     // Athens, GA
    val locs      = Array (LatLong (33.7490, -84.3880),             // 0. Atlanta, GA
                           LatLong (34.2979, -83.8241),             // 1. Gainesville, GA
                           LatLong (33.5957, -83.4679),             // 2. Madison, GA
                           LatLong (33.4735, -82.0105),             // 3. Augusta, GS
                           LatLong (32.8407, -83.6324),             // 4. Macon, GA
                           LatLong (33.0801, -83.2321))             // 5. Milledgeville, GA

     val nearest = sensorLoc.kNearest (locs)                        // find nearby locations

     banner (s"locations nearest to $sensorLoc")
     for ll <- nearest do println ((ll._1, ll._2*meters2Miles))

end latLongTest4

