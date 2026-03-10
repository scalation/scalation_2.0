
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Dec 11 18:35:14 EST 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Database: Teaching and TA Assignment
 */

package scalation
package database
package relation

import scala.collection.mutable.{ArrayBuffer => VEC}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Ex_Teaching` object contains table definitions for the `TASchedule` app.
 */
object Ex_Teaching:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The course table - for courses to be assigned TAs.
     *  dept:   department code
     *  cno:    course number (lab course 13011 means 1301L)
     *  title:  course title
     */
    val course = Relation ("course",
        Array ("dept", "cno", "title"),
        Array ('S', 'I', 'X'), Array ("cno"),
        VEC [Row] (Vector ("CSCI", 1100,  "Topics in Computing"),
                   Vector ("CSCI", 11001, "Topics in Computing Lab"),
                   Vector ("CSCI", 1210,  "Computer Modeling and"),
                   Vector ("CSCI", 1300,  "Intro to Python Programming"),
                   Vector ("CSCI", 13001, "Intro Python Programming Lab"),
                   Vector ("CSCI", 1301,  "Intro Computing and Program"),
                   Vector ("CSCI", 13011, "Intro Computing Program Lab"),
                   Vector ("CSCI", 1302,  "Software Development"),
                   Vector ("CSCI", 1360,  "Informatics and Data Analytics"),
                   Vector ("CSCI", 1730,  "Systems Programming"),
                   Vector ("CSCI", 2150,  "Intro Scientif Comp"),
                   Vector ("CSCI", 21501, "Lab Scientif Comp"),
                   Vector ("CSCI", 2610,  "Discrete Mathematics for CSCI"),
                   Vector ("CSCI", 2611,  "Discrete Mathematics for Engr"),
                   Vector ("CSCI", 2670,  "Intro to Theory of Computing"),
                   Vector ("CSCI", 2720,  "Data Structures"),
                   Vector ("CSCI", 2725,  "Data Structures Data Science"),
                   Vector ("CSCI", 3030,  "Computing Ethics and Society"),
                   Vector ("CSCI", 3360,  "Data Science I"),
                   Vector ("CSCI", 4050,  "Software Engineering"),
                   Vector ("CSCI", 4060,  "Mobile Software Development"),
                   Vector ("CSCI", 4150,  "Num Simu Sci and Eng"),
                   Vector ("CSCI", 4250,  "Cyber Security"),
                   Vector ("CSCI", 4300,  "Web Programming"),
                   Vector ("CSCI", 4360,  "Data Science II"),
                   Vector ("CSCI", 4370,  "Database Management"),
                   Vector ("CSCI", 4380,  "Data Mining"),
                   Vector ("CSCI", 4470,  "Algorithms"),
                   Vector ("CSCI", 4550,  "Artif Intelligence"),
                   Vector ("CSCI", 4690,  "Graph Theory"),
                   Vector ("CSCI", 4720,  "Computer Architecture Organiza"),
                   Vector ("CSCI", 4730,  "Operating Systems"),
                   Vector ("CSCI", 4760,  "Computer Networks"),
                   Vector ("CSCI", 4780,  "Distr Comp Systems"),
                   Vector ("CSCI", 4795,  "Cloud Computing"),
                   Vector ("CSCI", 4800,  "Human-Comp Interact"),
                   Vector ("CSCI", 4810,  "Computer Graphics"),
                   Vector ("CSCI", 4840,  "Signal Processing"),
                   Vector ("CSCI", 4850,  "Biomedical Image Analysis")))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The course-section table - for courses to be assigned TAs.
     *  crn:      course-section reference number
     *  cno:      course number
     *  period1:  main time periods
     *  period2:  additional time periods
     *  ilname:    name of the instructor
     *  seats:    maximum enrollment
     *  avail:    seats still available
     */
    val section = Relation ("section",
        Array ("crn", "cno", "period1", "period2", "ilname", "seats", "avail"),
        Array ('I', 'I', 'X', 'X', 'S', 'I', 'I'), Array ("crn"),
        VEC [Row] (Vector (23551, 1100,  "M W 01:50 pm-02:40 pm", "", "Vance", 72, 48),
                   Vector (26160, 11001, "M W 09:10 am-10:00 am", "", "Vance", 24, 19),
                   Vector (26221, 11001, "M W 10:20 am-11:10 am", "", "Vance", 24, 12),
                   Vector (26226, 11001, "T R 02:20 pm-03:35 pm", "", "Vance", 24, 17),

                   Vector (59910, 1210,  "M W F 01:50 pm-02:40 pm", "", "Hybinette", 42, 25),

                   Vector (56805, 1300,  "T R 11:10 am-12:25 pm", "", "Liang", 60, 29),
                   Vector (56806, 13001, "M W 09:10 am-10:00 am", "", "Liang", 30, 15),
                   Vector (56807, 13001, "M W 03:00 pm-03:50 pm", "", "Liang", 30, 14),

                   Vector (27812, 1310,  "T R 09:35 am-10:50 am", "", "Amirian", 120, 0),
                   Vector (27827, 1310,  "T R 11:10 am-12:25 pm", "", "Saleh", 96, 0),
                   Vector (48887, 1310,  "T R 02:20 pm-03:35 pm", "", "Liang", 90, 14),
                   Vector (49132, 1310,  "T R 03:55 pm-05:10 pm", "", "Liang", 90, 54),
                   Vector (26939, 13011, "M W 10:20 am-11:10 am", "", "Amirian", 30, 0),
                   Vector (26959, 13011, "M W 12:40 pm-01:30 pm", "", "Amirian", 30, 0),
                   Vector (26962, 13011, "M W 01:50 pm-02:40 pm", "", "Amirian", 30, 0),
                   Vector (26985, 13011, "T R 02:20 pm-03:35 pm", "", "Amirian", 30, 0),
                   Vector (26986, 13011, "M W 11:30 am-12:20 pm", "", "Saleh", 24, 0),
                   Vector (27832, 13011, "M W 03:00 pm-03:50 pm", "", "Saleh", 24, 0),
                   Vector (41173, 13011, "M W 04:10 pm-05:00 pm", "", "Saleh", 24, 0),
                   Vector (44510, 13011, "T R 09:35 am-10:50 am", "", "Saleh", 24, 0),
                   Vector (44511, 13011, "M W 09:10 am-10:00 am", "", "Liang", 30, 5),
                   Vector (45345, 13011, "M W 10:20 am-11:10 am", "", "Liang", 30, 0),
                   Vector (49023, 13011, "T R 03:55 pm-05:10 pm", "", "Liang", 30, 9),
                   Vector (49133, 13011, "M W 11:30 am-12:20 pm", "", "Liang", 30, 12),
                   Vector (49137, 13011, "M W 03:00 pm-03:50 pm", "", "Liang", 30, 12),

                   Vector (26245, 1302,  "M 08:00 am-08:50 am", " T R 08:00 am-09:15 am", "Cotterell", 85, 5),
                   Vector (26311, 1302,  "M 12:40 pm-01:30 pm", " T R 12:45 pm-02:00 pm", "Barnes", 85, 0),
                   Vector (36424, 1302,  "T R 02:20 pm-03:35 pm", " M 03:00 pm-03:50 pm", "Cotterell", 85, 0),
                   Vector (49139, 1302,  "M 03:00 pm-03:50 pm", " T R 02:20 pm-03:35 pm", "Barnes", 90, 0),

                   Vector (59926, 1360,  "T R 02:20 pm-03:35 pm", " M 03:00 pm-03:50 pm", "GharehMohammadi", 40, 20),

                   Vector (26313, 1730,  "W 12:40 pm-01:30 pm", " T R 11:10 am-12:25 pm", "Lamarca", 30, 0),
                   Vector (26326, 1730,  "F 09:10 am-10:00 am", " T R 11:10 am-12:25 pm", "Lamarca", 30, 1),
                   Vector (26337, 1730,  "T R 11:10 am-12:25 pm", " F 11:30 am-12:20 pm", "Lamarca", 30, 1),
                   Vector (26347, 1730,  "T R 11:10 am-12:25 pm", " F 01:50 pm-02:40 pm", "Lamarca", 30, 5),
                   Vector (38860, 1730,  "T R 03:55 pm-05:10 pm", " W 01:50 pm-02:40 pm", "Lamarca", 30, 0),
                   Vector (38861, 1730,  "F 10:20 am-11:10 am", " T R 03:55 pm-05:10 pm", "Lamarca", 30, 23),
                   Vector (44514, 1730,  "T R 03:55 pm-05:10 pm", " F 12:40 pm-01:30 pm", "Lamarca", 30, 18),
                   Vector (59927, 1730,  "T R 03:55 pm-05:10 pm", " F 03:00 pm-03:50 pm", "Lamarca", 30, 18),

                   Vector (26354, 2150,  "T R 09:35 am-10:50 am", "", "Taha", 48, 0),
                   Vector (52453, 2150,  "T R 03:55 pm-05:10 pm", "", "Hollingsworth", 60, 0),
                   Vector (26359, 21501, "M W 04:10 pm-05:00 pm", "", "Taha", 24, 0),
                   Vector (38493, 21501, "T R 11:10 am-12:25 pm", "", "Taha", 24, 0),
                   Vector (52455, 21501, "T R 09:35 am-10:50 am", "", "Hollingsworth", 30, 0),
                   Vector (52456, 21501, "T R 12:45 pm-02:00 pm", "", "Hollingsworth", 30, 0),

                   Vector (26368, 2610,  "T R 08:00 am-09:15 am", " W 08:00 am-08:50 am", "Peng", 90, 47),
                   Vector (26372, 2610,  "T R 02:20 pm-03:35 pm", " W 03:00 pm-03:50 pm", "Amirian", 90, 0),
                   Vector (43005, 2610,  "T R 03:55 pm-05:10 pm", " W 04:10 pm-05:00 pm", "Peng", 90, 0),

                   Vector (26432, 2611,  "M W F 10:20 am-11:10 am", "", "Amirian", 65, 26),

                   Vector (26481, 2670,  "T R 09:35 am-10:50 am", " M 10:20 am-11:10 am", "Peng", 44, 0),
                   Vector (38341, 2670,  "T R 11:10 am-12:25 pm", " M 11:30 am-12:20 pm", "Hollingsworth", 65, 0),
                   Vector (49154, 2670,  "T R 12:45 pm-02:00 pm", " M 12:40 pm-01:30 pm", "Hollingsworth", 71, 0),

                   Vector (26486, 2720,  "M 10:20 am-11:10 am", " T R 03:55 pm-05:10 pm", "Meena", 75, 19),
                   Vector (26490, 2720,  "T R 12:45 pm-02:00 pm", " M 12:40 pm-01:30 pm", "Arpinar", 75, 50),
                   Vector (43006, 2720,  "M 03:00 pm-03:50 pm", " T R 02:20 pm-03:35 pm", "Meena", 65, 0),

                   Vector (57505, 2725,  "T R 03:55 pm-05:10 pm", " M 04:10 pm-05:00 pm", "Amirian", 48, 22),

                   Vector (26497, 3030,  "M 12:40 pm-01:30 pm", " M W 11:30 am-12:20 pm", "Funk", 30, 0),
                   Vector (26510, 3030,  "M W 11:30 am-12:20 pm", " M 01:50 pm-02:40 pm", "Funk", 30, 0),
                   Vector (36665, 3030,  "F 09:10 am-10:00 am", " M W 11:30 am-12:20 pm", "Funk", 30, 0),
                   Vector (38119, 3030,  "M W 11:30 am-12:20 pm", " F 03:00 pm-03:50 pm", "Funk", 30, 0),

                   Vector (41607, 3360,  "M 04:10 pm-05:00 pm", " T R 03:55 pm-05:10 pm", "Li", 48, 9),

                   Vector (40774, 4050,  "M 10:20 am-11:10 am", " T R 09:35 am-10:50 am", "Saleh", 30, 7),
                   Vector (45016, 4050,  "M 11:30 am-12:20 pm", " T R 11:10 am-12:25 pm", "Meena", 44, 0),
                   Vector (46855, 4050,  "M 12:40 pm-01:30 pm", " T R 12:45 pm-02:00 pm", "Saleh", 35, 9),

                   Vector (49335, 4060,  "T R 09:35 am-10:50 am", " W 10:20 am-11:10 am", "Kochut", 42, 0),

                   Vector (40821, 4150,  "T R 12:45 pm-02:00 pm", " W 12:40 pm-01:30 pm", "Lamarca", 45, 32),

                   Vector (44365, 4250,  "W 04:10 pm-05:00 pm", " T R 03:55 pm-05:10 pm", "Guan", 40, 26),

                   Vector (26666, 4300,  "W 03:00 pm-03:50 pm", " T R 02:20 pm-03:35 pm", "Guimaraes", 65, 0),
                   Vector (52481, 4300,  "W 04:10 pm-05:00 pm", " T R 03:55 pm-05:10 pm", "Guimaraes", 65, 0),

                   Vector (48588, 4360,  "T R 12:45 pm-02:00 pm", " W 12:40 pm-01:30 pm", "Miller", 34, 1),

                   Vector (40880, 4370,  "T R 11:10 am-12:25 pm", " W 11:30 am-12:20 pm", "Guimaraes", 56, 0),

                   Vector (56828, 4380,  "T R 03:55 pm-05:10 pm", " W 04:10 pm-05:00 pm", "Liu", 30, 0),

                   Vector (41098, 4470,  "T R 11:10 am-12:25 pm", " M 11:30 am-12:20 pm", "Cai", 25, 2),

                   Vector (56830, 4550,  "T R 11:10 am-12:25 pm", "", "Van Orman", 17, 2),
                   Vector (56832, 4550,  "T R 02:20 pm-03:35 pm", "", "Van Orman", 19, 1),

// MATH            Vector (54438, 4690,  "M W F 09:10 am-10:00 am", Petridis", 6, 0),

                   Vector (26690, 4720,  "M 08:00 am-08:50 am", " T R 08:00 am-09:15 am", "Keshtgari", 75, 0),
                   Vector (41122, 4720,  "W 04:10 pm-05:00 pm", " T R 03:55 pm-05:10 pm", "Bhandarkar", 75, 54),

                   Vector (40899, 4730,  "W 03:00 pm-03:50 pm", " T R 02:20 pm-03:35 pm", "Wang", 50, 1),

                   Vector (40904, 4760,  "T R 09:35 am-10:50 am", " M 10:20 am-11:10 am", "Keshtgari", 73, 0),
                   Vector (46846, 4760,  "M 12:40 pm-01:30 pm", " T R 12:45 pm-02:00 pm", "Keshtgari", 75, 0),

                   Vector (40916, 4780,  "W 12:40 pm-01:30 pm", " T R 12:45 pm-02:00 pm", "Ramaswamy", 28, 2),

                   Vector (50148, 4795,  "W 10:20 am-11:10 am", " T R 09:35 am-10:50 am", "Kim", 40, 0),

                   Vector (44356, 4800,  "W 12:40 pm-01:30 pm", " T R 12:45 pm-02:00 pm", "Cotterell", 55, 0),

                   Vector (52483, 4810,  "T R 11:10 am-12:25 pm", " W 11:30 am-12:20 pm", "Arabnia", 60, 1),

                   Vector (59985, 4840,  "T R 09:35 am-10:50 am", " W 10:20 am-11:10 am", "Liu", 30, 25),

                   Vector (56870, 4850,  "T R 11:10 am-12:25 pm", " W 11:30 am-12:20 pm", "Liu", 30, 24)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The teaching assistants table.
     *  lname:  last name of TA
     *  fname:  first name of TA
     *  level:  MS 0, PhD 1, IoR 2, GSRA 3
     *  tep:    English Level 3, Level 4
     *  hours:  number of hours per week, e.g., 13.3 or 17.8
     */
    val ta = Relation ("ta",
        Array ("lname", "fname", "level", "tep", "hours"),
        Array ('S', 'S', 'I', 'I', 'D'), Array ("lname"),
        VEC [Row] (Vector ("Newton",   "Issac",  0, 3, 13.3),
                   Vector ("Einstein", "Albert", 1, 4, 17.8)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The class schdedule for the teaching assistants table.
     *  lname:    last name of TA
     *  periods:  the perioids for which the TA is taking classes
     *            71,72,73,74,75,76,77,78,,1,2,3,4,5,6,7,8,9,10
     *  cnos:     the course numbers as string, e.g., "6370, 6470"
     */
    val ta_sched = Relation ("ta_sched",
        Array ("lname", "fname", "periods", "cnos"),
        Array ('S', 'S', 'S', 'S'), Array ("lname"),
        VEC [Row] (Vector ("?", "?", "", "9000,9300"),
                   Vector ("?", "?", "TR72,TR75,W3,M7", "6360,6950,7007,8955,9000")))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The preferred teaching assistants per faculty member x course.
     *  ilname:   instructor name
     *  cno:      course number
     *  lname1:   last name of TA - first choice
     *  lname2:   last name of TA - second choice
     */
    val i_pref = Relation ("i_pref",
        Array ("ilname", "cno", "lname1", "lname2"),
        Array ('S', 'I', 'S', 'S'), Array ("ilname"),
        VEC [Row] (Vector ("?", 1301, "?", "?"),
                   Vector ("?", 1302, "?", "?")))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The assignment of teaching assistants to course sections (crn) table.
     *  lname:   last name of TA
     *  fname:   first name of TA
     *  cno:     course name
     *  crn:     course-section reference number (-1 => delegated to instructor)
     *  hours:   hours per week assigned the instructor-course
     *  ilname:  instructor name
     */
    val ta_assign = Relation ("ta_assign",
        Array ("lname", "fname", "cno", "crn", "hours", "ilname"),
        Array ('S', 'S', 'I', 'I', 'D', 'S'), Array ("lname"),
        VEC [Row] (Vector ("?", "?", 2670, -1, 17.8, "?"),
                   Vector ("?", "?", 1730, -1, 13.3, "?")))

end Ex_Teaching

import Ex_Teaching._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `showTables` main function prints all the tables defined in the `Teachning` object.
 *  > runMain scalation/database.relation.showTables
 */
@main def showTables (): Unit =

    println (s"course: rows = ${course.rows}")
    course.show ()

    println (s"section: rows = ${section.rows}")
    section.show ()

    println (s"ta: rows = ${ta.rows}")
    ta.show ()

    println (s"ta_sched: rows = ${ta_sched.rows}")
    ta_sched.show ()

    println (s"i_pref: rows = ${i_pref.rows}")
    i_pref.show ()

    println (s"ta_assign: rows = ${ta_assign.rows}")
    ta_assign.show ()

    section.project ("ilname").show ()

/*
    val detail = ta_assign.join (VEC ("ilname", "cno"), VEC ("ilname", "cno"), section)
                          .project ("lname", "fname", "cno", "hours", "ilname", "crn2", "period1") 
    println (s"detail: rows = ${detail.rows}")
    detail.show ()

    (ta_assign.project ("lname") minus detail.project ("lname")).show ()
*/

end showTables

