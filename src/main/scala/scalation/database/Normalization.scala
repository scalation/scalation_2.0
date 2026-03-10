
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Apr  2 15:37:57 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @see     https://github.com/scalation/scalation_2.0
 *
 *  @note    Support Functions for Database Normalization Based on Functional Dependencies (FDs)
 *           Includes implementations of the BCNF Decomposition Algorithm
 *                                   and the 4NF  Decomposition Algorithm
 *                                   and the 3NF  Synthesis Algorithm
 *
 *  Relation r(R) defined of Schema R
 *  Assume X, Y ⊆ R
 *
 *  Functional Dependency (FD)  X -> Y
 *           for any t, u in r,  t[X] = u[X]  =>  t[Y] = u[Y]
 *
 *  Multi-Valued Dependency (MVD)  X ->> Y
 *           assuming R = XYZ, whenever (x, y1, z1) and (x, y2, z2) in r  =>
 *                                      (x, y1, z2) and (x, y2, z1) must also be in r
 *
 *  BCNF     Boyce-Codd Normal Form -- for all nontrivial X -> Y, X must be a superkey
 *  4NF      Fourth Normal Form     -- for all nontrivial X ->> Y, X must be a superkey
 *  3NF      Third Normal Form      -- for all nontrivial X -> Y, X must be a superkey OR
 *                                  -- Y must consist of prime attributes
 *
 *  FIX: (1) improve 4NF Decomposition, (2) finish Chase Algorithm that handles both FDs and MVDs,
 *       (3) remove the current limit for `Tableau` on the number of attributes, currently 26 "A" to "Z".
 *       (4) remove "TreeSet" prefix when printing `Attr` without constantly calling the `cut` method
 */

package scalation
package database

import scala.collection.mutable.{ArrayBuffer => VEC, SortedSet => SET}
//import scala.runtime.ScalaRunTime.stringOf
import scala.util.boundary, boundary.break

// @note:  &~ means set difference (alias for diff)

type Attrs   = SET [Char]                                          // type for set of attributes
type Symbols = Array [String]                                      // types for array of symbols (@see `Tableau`)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Cut out "TreeSet" from the `Attrs` toString method.
 *  @param a  the given set of attributes
 */
def cut (a: Attrs): String = 
    val sa = a.toString
    val n  = sa.indexOf ('(')
    sa.drop (n)
end cut

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Cut out the prefix (e.g., "ArrayBuffer") from the `Attrs` toString method.
 *  @param a  the given collection of sets of attributes
 */
def cut (a: VEC [Attrs]): String = 
    val sa = a.toString
    val n  = sa.indexOf ('(')
    sa.drop (n)
end cut


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FD` case class is used to represent a Functional Dependency (x -> y).
 *  Caveat:  Can also be be used for a Multi-Valued Dependency (MVD) x ->> y as well
 *  @param x  the Left-Hand Side (LHS) attributes of the FD
 *  @param y  the Right-Hand Side (RHS) attributes of the FD
 */
case class FD (x: Attrs, y: Attrs):

    override def toString: String = s"${x.toString.drop (7)} -> ${y.toString.drop (7)}"

end FD


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Normalization` class provides methods useful in normalization theory
 *  that can be used to assist in designing relational databases.
 *  These include Closure, Superkey, Key, Losslessness, Dependency, Preservation,
 *  BCNF Decomposition, 4NF Decomposition, Minimal Cover, and 3NF Synthesis.
 *  @see `scalation.SetExt`, scalation.database.BinTree`
 *  @param r   the schema or complete set of attributes R
 *  @param fd  the given collection of Functional Dependencies (FDs)
 */
class Normalization (r: Attrs, fd: VEC [FD]):

    private val debug = debugf ("Normalization", true)             // the debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the closure of z (Z+), i.e., given the attributes in set z, what
     *  attributes in R are functionally dependent on z.
     *  @param z    the given set of attributes
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def closure (z: Attrs, fd_ : VEC [FD] = fd): Attrs =
        val zp = z.clone
        var changes = true
        while changes do
            changes = false
            for f <- fd_ if f.x ⊆ zp && f.y ⊈ zp do { zp ++= f.y; changes = true }
        zp
    end closure

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the restricted closure of z (Z+_) with respect to sub-schema ri.
     *      Z+_ = (Z ∩ Ri)+ ∩ Ri
     *  @param z    the given set of attributes
     *  @param ri   the given sub-schema (subset of r)
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def rclosure (z: Attrs, ri: Attrs, fd_ : VEC [FD] = fd): Attrs =
        closure (z ∩ ri, fd_) ∩ ri

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the restricted closure of z (Z+_p) with respect to db design (set of tables)
     *  p = {r1, r2, ...rk}.
     *  @param z    the given set of attributes
     *  @param p    the set of sub-schemas defining the tables
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def rclosure_p (z: Attrs, p: VEC [Attrs], fd_ : VEC [FD] = fd): Attrs =
        val zp = z.clone                                           // attributes derived from z using p
        var changes = true
        while changes do
            changes = false
            for ri <- p do                                         // use each table ri in p
                val zi = rclosure (zp, ri, fd_)
                if zi ⊈ zp then { zp ++= zi; changes = true }
                debug ("rclosure_p", s"from $ri: zp = $zp")
        zp
    end rclosure_p

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the functional dependency f is preserved by db design p,
     *  i.e., Y ⊆ X+_p.
     *  @param f    the given function dependency
     *  @param p    the set of sub-schemas defining the tables
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def preserve_ (f: FD, p: VEC [Attrs], fd_ : VEC [FD] = fd): Boolean =
        f.y ⊆ rclosure_p (f.x, p, fd_)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the functional dependency f is preserved by db design p,
     *  i.e., Y ⊆ X+_p.  Combines last two methods for early algorithm termination.
     *  @param f    the given function dependency
     *  @param p    the set of sub-schemas defining the tables
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def preserve (f: FD, p: VEC [Attrs], fd_ : VEC [FD] = fd): Boolean = boundary:
        val zp = f.x.clone                                         // attributes derived from f.x using p
        var changes = true
        while changes do
            changes = false
            for ri <- p do                                         // use each table ri in p
                val zi = rclosure (zp, ri, fd_)
                if zi ⊈ zp then { zp ++= zi; changes = true }
                debug ("preserve", s"from Ri = $ri: zp = $zp")
                if f.y ⊆ zp then break (true)
        false 
    end preserve

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the attributes z form a superkey for sub-schema ri.
     *  @param z    the given set of attributes
     *  @param ri   the given sub-schema (a subset of r, defaults to r)
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def superkey (z: Attrs, ri: Attrs = r, fd_ : VEC [FD] = fd): Boolean =
        ri ⊆ closure (z, fd_)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the attributes z form a key for sub-schema ri.
     *  @param z    the given set of attributes
     *  @param ri   the given sub-schema (a subset of r, defaults to r)
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def key (z: Attrs, ri: Attrs = r, fd_ : VEC [FD] = fd): Boolean =
        if ! superkey (z, ri, fd_) then return false
        z.∄ ( (a: Char) => superkey (z &~ SET (a), ri, fd_) )
    end key

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find a (global) key for relational schema R efficiently using
     *  Algorithm based on Section 3 with modifications of
     *  An Efficient Algorithm to Compute the Candidate Keys of a Relational Database Schema.
     *  @author  Charles Moseley (of code)
     *  @see https://people.eecs.ku.edu/~hossein/Pub/Journal/1996-Saiedian-TCJ.pdf
     *  @param fd_mc  the minimal cover (mc) set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def findKey (fd_mc : VEC [FD] = fd): Attrs = boundary:
                                                                   // Step 1: form sets nrhs, bhs
        val lhs = SET [Char] ()                                    // LHS attributes
        val rhs = SET [Char] ()                                    // RHS attributes
        for f <- fd_mc do { lhs ++= f.x; rhs ++= f.y }             // collect sets of LHS and RHS attributes
        val nrhs = r &~ rhs                                        // attributes not in any RHS
        val bhs  = lhs ∩ rhs                                       // attributes in both LHS and RHS
        debug ("findKey", s"bhs = $bhs, nrhs = $nrhs")

        if closure (nrhs, fd_mc) == r then                         // Step 2: is the attribute set nrhs a key?
            debug ("findKey", s"return since nrhs = $nrhs is a key")
            break (nrhs)                                           // found that nrhs is a key, return skipping Step 3

        for n <- 1 to bhs.size do                                  // Step 3: start with nrhs and try adding subsets of bhs
            for ss <- bhs.subsets (n) do                           // subsets of bhs of length n
                val key = nrhs ++ ss                               // form a possible key
                if closure (key, fd_mc) == r then                  // is it a key?
                    debug ("findKey", s"return since key = $key is a key")
                    break (key)                                    // found a key, return

        SET [Char] ()                                              // empty set => no key found
    end findKey

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the two sub-schemas { r1, r2 } represent a lossless decomposition
     *  using the Pairwise Losslessness Test (PLT).
     *  @param r1   the first sub-schema defining a table
     *  @param r2   the second sub-schema defining a table
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def lossless (r1: Attrs, r2: Attrs, fd_ : VEC [FD] = fd): Boolean =
        val z = closure (r1 ∩ r2, fd_)                             // take the closure of the intersection
        r1 ⊆ z || r2 ⊆ z
    end lossless

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the db design p is lossless by checking that one of its
     *  sub-schemas contains a global key.
     *  @param p    the set of sub-schemas defining the tables
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def lossless_ (p: VEC [Attrs], fd_ : VEC [FD] = fd): Boolean =
        p.exists (r ⊆ rclosure_p (_, p, fd_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the db design p is lossless by applying the chase algorithm
     *  for FDs (equality generating dependencies).  When the LHSs of two rows in
     *  a tableau agree, make their RHSs the same (i.e., make their symbols the same). 
     *  Given FD  x ->  y and two rows (x, y1, z1) and (x, y2, z2) update the second row to (x, y1, z2).
     *  If one of the symbols is distinguished, make both distinguished.
     *  @param tbl  the given tableau
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def chaseFD (tbl: Tableau, fd_ : VEC [FD] = fd): Unit =
        var changes = true
        while changes do
            changes = false
            for f <- fd_ do changes = tbl.equateSymbols (f)
    end chaseFD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the db design p is lossless by applying the chase algorithm
     *  for MVDs (tuple generating dependencies),
     *  Given MVD x ->> y and two rows (x, y1, z1) and (x, y2, z2) add a new row (x, y1, z2).
     *  @param tbl  the given tableau
     *  @param mvd  the set of Multi-Valued Dependencies (MVDs) to use
     */
    def chaseMVD (tbl: Tableau, mvd: VEC [FD]): Unit =
        var changes = true
        while changes do
            changes = false
            for f <- mvd do changes = tbl.genTuples (f)
            changes = false                                   // FIX - infinite loop otherwise
    end chaseMVD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the db design p is lossless by applying the chase algorithm
     *  for FDs (equality generating dependencies) and MVDs (tuple generating dependencies).
     *  @see `Tableau`
     *  @param p    the set of sub-schemas defining the tables
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     *  @param mvd  the set of Multi-Valued Dependencies (MVDs) to use
     */
    def chase (p: VEC [Attrs], fd_ : VEC [FD] = fd, mvd: VEC [FD] = null): Boolean =
        val tbl = new Tableau (r, p)
        banner (s"The initial tableau for database design p = ${cut(p)}")

        if fd_ != null then chaseFD (tbl, fd_)
        banner (s"The post FD tableau for database design p = ${cut(p)}")

        if mvd != null then chaseMVD (tbl, mvd)
        val lossless = tbl.allDistinguished ()
        banner (s"The final tableau for database design p = ${cut(p)} is lossless = $lossless")
        tbl.printTableau ()
        lossless
    end chase

//  B C N F

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the first Functional Dependency (FD) explicitly in fd_ that violates
     *  the BCNF rule that the LHS of all applicable, nontrivial FDs are superkeys.
     *  Return null if none found.
     *  @param ri   the i-th sub-table Ri
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def find_not_bcnf (ri: Attrs, fd_ : VEC [FD] = fd): FD = boundary:
        for f <- fd_ if f.x ⊆ ri do                                // LHS must be in Ri
            if ! superkey (f.x, ri, fd_) then                      // not a superkey w.r.t. Ri (violates BCNF)
                val y = f.y ∩ ri                                   // y is relevant RHS of f.y in Ri
                if y.nonEmpty then break (FD (f.x, y))             // return FD: f.x -> y
        null                                                       // none found
    end find_not_bcnf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the first Functional Dependency (FD) in (fd_)+, the closure, that violates
     *  the BCNF rule that the LHS of all applicable, nontrivial FDs are superkeys.
     *  Return null if none found.
     *  @param ri   the i-th sub-table Ri
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def find_not_bcnf_Fp (ri: Attrs, fd_ : VEC [FD] = fd): FD = boundary:
        val k = ri.size
        for x <- ri.subsets () if x.size in (1, k-1) do            // consider possible LHSs
            val y = rclosure (x, ri, fd_) &~ x                   // nontrivial FD x -> y inside Ri
            if y.nonEmpty && ! superkey (x, ri, fd_) then          // y is nonempty and x is not a superkey w.r.t Ri
                break (FD (x, y))                                  // return FD: x -> y
        null                                                       // none found
    end find_not_bcnf_Fp

    private [database] val bcnf_root = new BinTree [Attrs] (r)     // root of the BCNF Decomposition Tree

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** BCNF DECOMPOSITION ALGORITHM that builds a BCNF decomposition tree.
     *  Caveat: Comment out find_not_bcnf_Fp line to only consider FDs in F.
     *          Reorder FDs so that those with prime attributes come last.
     *  @param tree  the tree (sub-tree) to work off of (defaults to bcnf_root)
     *  @param ri    the i-th sub-table (defaults to r)
     *  @param fd_   the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def bcnf_decomp (tree: BinTree [Attrs] = bcnf_root, ri: Attrs = r, fd_ : VEC [FD] = fd): Unit =
        var f = find_not_bcnf (ri, fd_)                            // find FD in F that violates BCNF
        if f == null then f = find_not_bcnf_Fp (ri, fd_)           // find FD in F+ that violates BCNF <<<<
        println (s"use FD $f to decompose $ri")

        if f != null then
            val (r1, r2) = (f.x ∪ f.y, ri &~ f.y)
            println (s"decompose $ri into ($r1, $r2)")
            val tleft = tree.addLeft (r1)                          // add r1 as left child
            val trigh = tree.addRigh (r2)                          // add r2 as right child
            if r1.size > 2 then bcnf_decomp (tleft, r1, fd_)       // recursive call on left sub-tree
            if r2.size > 2 then bcnf_decomp (trigh, r2, fd_)       // recursive call on right sub-tree
    end bcnf_decomp

//  4 N F

    private [database] val _4nf_root = new BinTree [Attrs] (r)     // root of the 4NF Decomposition Tree

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** 4NF DECOMPOSITION ALGORITHM that builds a 4NF decomposition tree.
     *  Caveat:  Reorder FDs so that those with prime attributes come last.
     *  FIX - order of FD and MVD can be important - defer when prime on RHS
     *  @param mvd   the set of Multi-Valued Dependencies (MVDs) to use
     *  @param tree  the tree (sub-tree) to work off of (defaults to _4nf_root)
     *  @param ri    the i-th sub-table (defaults to r)
     *  @param fd_   the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def _4nf_decomp (mvd: VEC [FD], tree: BinTree [Attrs] = _4nf_root, ri: Attrs = r, fd_ : VEC [FD] = fd): Unit =
        val deps = fd_ ++ mvd
        bcnf_decomp (tree, ri, deps)
    end _4nf_decomp

//  3 N F

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Shrink the LHS of each Functional Dependency (FD) that contains extraneous attributes.
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def shrink_LHS (fd_ : VEC [FD] = fd): Unit =
        for f <- fd_ do                                            // check each FD for extraneous attributes
            var changes = true
            while changes do
               changes = false
               for b <- f.x if f.y ⊆ closure (f.x &~ SET (b)) do 
                   f.x -= b                                        // remove extraneous attribute B
                   changes = true
    end shrink_LHS

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a reduced set of Functional Dependencies (FDs) with the redundant ones removed. 
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def remove_redundant_FDs (fd_ : VEC [FD] = fd): VEC [FD] =
        val rfd = fd_.clone ()                                     // to hold the reduced set of FDs
        var i = 0
        while i < rfd.size do
            val f_i = fd_(i)                                       // i-th candidate FD to check
            if f_i.y ⊆ closure (f_i.x, rfd diff VEC (f_i)) then
                rfd.remove (i)                                     // remove this redundant FD
            else
                i += 1                                             // move to the next FD
        rfd
    end remove_redundant_FDs

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a Minimal/Canonical Cover of Functional Dependencies (FDs) that are
     *  streamlined, yet equivalent to the give set fd_.
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def minimal_cover (fd_ : VEC [FD] = fd): VEC [FD] =
        var mcfd = VEC [FD] ()                                     // to hold minimal cover (mc) of FDs
        for f <- fd_; a <- f.y do mcfd += FD (f.x, SET (a))        // reform FDs to have singleton RHS
        println (s"STEP 1: mcfd = $mcfd")
        shrink_LHS (mcfd)                                          // eliminate extraneous attributes from LHSs
        println (s"STEP 2: mcfd = $mcfd")
        mcfd = remove_redundant_FDs (mcfd)                         // remove redundant FDs
        println (s"STEP 3: mcfd = $mcfd")
        mcfd
    end minimal_cover

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Merge the Functional Dependences (FDs) that share a common LHS.
     *  @param fd_  the set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def merge_fds (fd_ : VEC [FD] = fd): VEC [FD] =
        val mfd = VEC [FD] ()
        for (k, v) <- fd_.groupBy (_.x) do                         // group by the LHS (_.x)
            val vv = SET [Char] ()
            for f <- v do vv ++= f.y                               // collect the RHSs
            mfd += FD (k, vv)                                      // add the newly formed FD
        mfd                                                        // return merged FDs
    end merge_fds

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the collection of tables that are subsets of any other table in p.
     *  Note, removal of a subset table may mean that a CHECK constraint (rather
     *  than a KEY constraint) will be required to enforce the affected FD.
     *  @param p  the set of sub-schemas defining the tables
     */
    def subset_tables (p: VEC [Attrs]): VEC [Attrs] =
        val rem = VEC [Attrs] ()                                   // collection of subset tables
        for i <- p.indices do
            val p_i = p(i)                                         // candidate table to check
            for j <- p.indices if j != i && p_i ⊆ p(j) do rem += p_i
        rem
    end subset_tables

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** 3NF SYNTHESIS ALGORITHM that synthesizes tables from Functional Dependencies (FDs).
     *  Caveat: Assumes the Functional Dependencies (FDs) constitute a MINIMAL/CANONICAL COVER.
     *  @param fd_mc  the minimal cover (mc) set of Functional Dependencies (FDs) to use (defaults to fd)
     */
    def _3nf_synthesis (fd_mc: VEC [FD] = fd): VEC [Attrs] =
        val mfd = merge_fds (fd_mc)                                // STEP 1: MERGE FDs with common LHS
        val p   = VEC [Attrs] ()
        for f <- mfd do p += f.x ∪ f.y                             // STEP 2: each FD FORMS a table

        p --= subset_tables (p)                                    // STEP 3: REMOVE all SUBSET tables

        if lossless_ (p, fd_mc) then                               // STEP 4: LOSSLESS? need Ri+ = R for some i
            println ("design p is lossless => done as is")
        else
            println ("lossy => add a global key as a new table")   // use findKey to find a global key
            p += findKey (fd_mc)
        p
    end _3nf_synthesis

end Normalization


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Tableau` class support the creation is tableaux, i.e., abstract tables
 *  containing rows of symbols (distinguished and non-distinguished).
 *  @param r  the complete set of attributes 
 *  @param p  the set of sub-schemas defining the tables
 */
class Tableau (r: Attrs, p: VEC [Attrs]):

    private val debug = debugf ("Tableau", true)                    // the debug function
    private val _A    = 'A'.toInt                                   // integer values of letter A
    private val _a    = 'a'.toInt                                   // integer values of letter a
    private val m     = p.size                                      // number of sub-schema/rows
    private val n     = r.size                                      // number of attributes/columns
    private val tabl  = new VEC [Symbols] (m)                       // storage for two-dimensional abstract table

    for i <- p.indices do tabl += makeRow (p(i), i)                 // add all rows to the tableau

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make a row for Tableau tabl, the abstract table of strings, where for row i, element j
     *  is a distinguished (e.g., b) symbols when character j is in sub-relation p_i,
     *  and is otherwise a non-distinguished symbol (e.g., b3).
     *  @param ri  the cuurent sub-schema
     *  @param i   the i-th row
     */
    def makeRow (ri: Attrs, i: Int): Symbols =
        val row = Array.ofDim [String] (n)
        for j <- 0 until n do
            val ch: Char = (_A + j).toChar
            val dsym = (_a + j).toChar.toString                // distinguished symbol
            row(j) = if ri contains ch then dsym else dsym + i
        row
    end makeRow

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make a new row for Tableau tabl by crossing row i with row j as follows:
     *  Given MVD x ->> y and two rows (x, y1, z1) and (x, y2, z2) add a new row (x, y2, z1).
     *  @param x  the LHS attributes of the MVD
     *  @param y  the RHS attributes of the MVD
     *  @param i  the first row
     *  @param h  the second row
     */
    def makeRow (x: Attrs, y: Attrs, i: Int, j: Int): Symbols =
        println (s"makeRow based on MVD: $x ->> $y")
        val row = Array.ofDim [String] (n)
        for k <- 0 until n do
            val ch: Char = (_A + k).toChar
            row(k) = if y contains ch then tabl(j)(k) else tabl(i)(k)
        row
    end makeRow

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Project the given row in the tableau onto attributes x".
     *  @param x    the given attributes
     *  @param row  the given row in the tableau
     */
    def onto (x: Attrs, row: Symbols): Symbols =
//      debug ("onto", s"project row = ${stringOf (row)} onto attributes x = $x")
        (for ai <- x yield { val i = ai - _A; row(i) }).toArray
    end onto

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Equate RHS symbols for attributes/columns in the tableau for rows i and j.
     *  @param y  the attributes to equate (make the same)
     *  @param i  index of i-th row
     *  @param j  index of j-th row
     */
    def equateRHS (y: Attrs, i: Int, j: Int): Boolean =
        val same = onto (y, tabl(i)) sameElements onto (y, tabl(j))
        if ! same then
             for ak <- y do
                 val k = ak - _A
                 if tabl(i)(k) < tabl(j)(k) then tabl(j)(k) = tabl(i)(k)
                 else tabl(i)(k) = tabl(j)(k)
        end if
        ! same
    end equateRHS

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For Functional Dependency (FD) f, try to equate symbols in the tableau
     *  by comparing each row with every other row.  When two rows agree on their LHSs,
     *  make their RHSs the same.  Return whether any symbols were changed.
     *  @param f  the FD to use to try to equate symbols
     */
    def equateSymbols (f: FD): Boolean =
        printTableau ()
        debug ("equateSymbols", s"use FD $f to equate symbols in the tableau")
        var change = false
        for i <- tabl.indices; j <- 0 until i if j != i do
            if onto (f.x, tabl(i)) sameElements onto (f.x, tabl(j)) then
                println (s"for f.x = ${f.x}: rows i = $i and j = $j have same LHS")
                if equateRHS (f.y, i, j) then change = true
        end for
        change
    end equateSymbols

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For Multi-Valued Dependency (MVD) f, try to generate and add new rows into the tableau
     *  by comparing each row with every other row.  When two rows agree on their LHSs,
     *  make their RHSs the same.  Return whether any symbols were changed.
     *  @param f  the MVD to use to try to equate symbols
     */
    def genTuples (f: FD): Boolean =
        printTableau ()
        debug ("genTuples", s"use MVD $f to generate tuples add into the tableau")
        var change = false
        val nrows  = tabl.size
        for i <- 0 until nrows; j <- 0 until i if j != i do
            if onto (f.x, tabl(i)) sameElements onto (f.x, tabl(j)) then
                println (s"for f.x = ${f.x}: rows i = $i and j = $j have same LHS")
                val newRow = makeRow (f.x, f.y, i, j)
                if ! (newRow sameElements tabl(i)) then
                     tabl += newRow
                     change = true
            end if
        end for
        change
    end genTuples

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the final tableau contains a row of all distinguished symbols.
     *  Distinguished symbols "a" to "z".  Non-distinguished symbols "a1" to "z25" where
     *  e.g. 25 is the number of rows in the tableau.
     */
    def allDistinguished (): Boolean =
        cfor (0, m) { i =>
            if tabl(i).forall (_.length == 1) then
                debug ("allDistinguished", s"row i = $i of tableau has all distinguished symbols")
                return true
        } // cfor
        false
    end allDistinguished

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the tableau.
     */
    def printTableau (): Unit =
        println ("-" * (27 + 3 * n))
        println (f"| Tableau              | ${cut (r)} |")
        println ("-" * (27 + 3 * n))
        for i <- 0 until m do
            print (f"| ${cut (p(i))}%20s |")
            for j <- 0 until n do print (f"${tabl(i)(j)}%3s")
            println ("  |")
        println ("-" * (27 + 3 * n))
    end printTableau

end Tableau


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `normalizationTest` main method tests the `Normalization` class.
 *  For this example all FD are preserved.
 *  > runMain scalation.database.normalizationTest
 */
@main def normalizationTest (): Unit =

    val r  = SET ('A', 'B', 'C', 'D')                              // R = ABCD
    val fd = VEC (FD (SET ('A', 'B'), SET ('C')),                  // F = AB -> C
                  FD (SET ('A', 'D'), SET ('B', 'C')))             //     AD -> BC

    val db = Normalization (r, fd)

    banner ("Schema r and Functional Dependencies fd")
    println (s"r  = $r")
    println (s"fd = $fd")

    banner ("Closure")
    println (s"closure (SET ('A', 'B')) = ${db.closure (SET ('A', 'B'))}")
    println (s"closure (SET ('A', 'D')) = ${db.closure (SET ('A', 'D'))}")

    val r1 = SET ('A', 'B', 'C')
    banner (s"Restricted Closure for r1 = $r1")
    println (s"rclosure (SET ('A', 'B'), r1) = ${db.rclosure (SET ('A', 'B'), r1)}")
    println (s"rclosure (SET ('A', 'D'), r1) = ${db.rclosure (SET ('A', 'D'), r1)}")

    val r2 = SET ('A', 'B', 'D')
    banner (s"Restricted Closure for r2 = $r2")
    println (s"rclosure (SET ('A', 'B'), r2) = ${db.rclosure (SET ('A', 'B'), r2)}")
    println (s"rclosure (SET ('A', 'D'), r2) = ${db.rclosure (SET ('A', 'D'), r2)}")

    val p = VEC (r1, r2)
    banner (s"Restricted Closure for p = $p}")
    println (s"rclosure_p (SET ('A', 'B'), p) = ${db.rclosure_p (SET ('A', 'B'), p)}")
    println (s"rclosure_p (SET ('A', 'D'), p) = ${db.rclosure_p (SET ('A', 'D'), p)}")

    banner (s"Is FD Preserved?")
    println (s"preserve (FD (SET ('A', 'B'), SET ('C'), p) = ${db.preserve (FD (SET ('A', 'B'), SET ('C')), p)}")
    println (s"preserve (FD (SET ('A', 'D'), SET ('B'), p) = ${db.preserve (FD (SET ('A', 'D'), SET ('B')), p)}")
    println (s"preserve (FD (SET ('A', 'D'), SET ('C'), p) = ${db.preserve (FD (SET ('A', 'D'), SET ('C')), p)}")

    banner (s"LHS of FD a Superkey?")
    println (s"superkey (SET ('A', 'B')) = ${db.superkey (SET ('A', 'B'))}")
    println (s"superkey (SET ('A', 'D')) = ${db.superkey (SET ('A', 'D'))}")

    banner (s"LHS of FD a Key?")
    println (s"key (SET ('A', 'B')) = ${db.key (SET ('A', 'B'))}")
    println (s"key (SET ('A', 'D')) = ${db.key (SET ('A', 'D'))}")

    banner ("Lossless?")
    println (s"lossless (r1, r2) = ${db.lossless (r1, r2)}")

end normalizationTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `normalizationTest2` main method tests the `Normalization` class.
 *  For this example some FDs may not be preserved.
 *
 *  fd = sid -> sname addr phone          A -> BCD
 *       cid -> cname desc hours pid      E -> FGHI
 *       pid -> pname rank                I -> JK
 *       sid cid -> grade                 AE -> L
 *  
 *  Comment out line (1) => success
 *  Comment out line (2) => Dependency Preservation fails
 *  Comment out lines (1 and 2) => Losslessness and Dependency Preservation fail
 *
 *  > runMain scalation.database.normalizationTest2
 */
@main def normalizationTest2 (): Unit =

    val r  = SET ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L')
    val fd = VEC (FD (SET ('A'), SET ('B', 'C', 'D')),
                  FD (SET ('E'), SET ('F', 'G', 'H', 'I')),        // what happens unpon swap
                  FD (SET ('I'), SET ('J', 'K')),                  // FD order
                  FD (SET ('A', 'E'), SET ('L')))
    val p  = VEC (SET ('A', 'B', 'C', 'D'),
                  SET ('E', 'F', 'G', 'H', 'I'),
                  SET ('E', 'J', 'K'),                             // (1) a BCNF Decomposition
//                SET ('I', 'J', 'K'),                             // (2) 3NF Synthesis
                  SET ('A', 'E', 'L'))

    val db = Normalization (r, fd)

    banner ("Schema r, Functional Dependencies fd, and DB Design p")
    println (s"r  = $r")
    println (s"fd = $fd")
    println (s"p  = $p")

    banner ("LHS Superkey?")
    for f <- fd do
        val x = f.x
        println (s"f = $f, closure ($x) = ${db.closure (x)}, " +
                        s"superkey ($x) = ${db.superkey (x)}")

    banner ("FD Violating BCNF?")
    println (s"find_not_bcnf (r) = ${db.find_not_bcnf (r)}")

    banner ("BCNF Decomposition?")
    db.bcnf_decomp ()
    db.bcnf_root.printTree ()

    banner ("Lossless?")
    println (s"lossless_ (p) = ${db.lossless_ (p)}")
    db.chase (p, fd)

    banner ("Dependency Preservation?")
    for f <- fd do
        println (s"f = $f, preserve (f, p) = ${db.preserve (f, p)}")

    banner ("Find a Global Key")
    println (s"findKey () = ${db.findKey ()}")

end normalizationTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `normalizationTest3` main method tests the `Normalization` class.
 *  For this example some FDs may not be preserved.  Also, has an MVD.
 *
 *  fd = sid -> sname addr phone          A -> BCD
 *       pid -> pname rank                I -> JK
 *       cid -> cname desc hours pid      E -> FGHI
 *       sid cid -> grade                 AE -> L
 *       did -> dname office              M -> NO
 *
 *  mvd = pid ->> did                     I ->> M
 *
 *  Comment out line (1) => success
 *  Comment out line (2) => Dependency Preservation fails
 *  Comment out lines (1 and 2) => Losslessness and Dependency Preservation fail
 *
 *  > runMain scalation.database.normalizationTest3
 */
@main def normalizationTest3 (): Unit =

    val r   = SET ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O')
    val fd  = VEC (FD (SET ('A'), SET ('B', 'C', 'D')),
                  FD (SET ('I'), SET ('J', 'K')),                  // FD order
                  FD (SET ('E'), SET ('F', 'G', 'H', 'I')),        // what id swapped
                  FD (SET ('A', 'E'), SET ('L')),
                  FD (SET ('M'), SET ('N', 'O')))
    val mvd = VEC (FD (SET ('I'), SET ('M')))
    val p   = VEC (SET ('A', 'B', 'C', 'D'),
                  SET ('E', 'F', 'G', 'H', 'I'),
                  SET ('E', 'J', 'K'),                             // (1) a BCNF Decomposition
//                SET ('I', 'J', 'K'),                             // (2) 3NF Synthesis
                  SET ('A', 'E', 'L'))

    val db = Normalization (r, fd)

    banner ("Schema r, Functional Dependencies fd, and DB Design p")
    println (s"r  = $r")
    println (s"fd = $fd")
    println (s"p  = $p")

    banner ("LHS Superkey?")
    for f <- fd do
        val x = f.x
        println (s"f = $f, closure ($x) = ${db.closure (x)}, " +
                        s"superkey ($x) = ${db.superkey (x)}")

    banner ("FD Violating BCNF?")
    println (s"find_not_bcnf (r) = ${db.find_not_bcnf (r)}")

    banner ("BCNF Decomposition?")
    db.bcnf_decomp ()
    db.bcnf_root.printTree ()

    banner ("Lossless?")
    println (s"lossless_ (p) = ${db.lossless_ (p)}")
    db.chase (p, fd, mvd)

    banner ("Dependency Preservation?")
    for f <- fd do
        println (s"f = $f, preserve (f, p) = ${db.preserve (f, p)}")

    banner ("Find a Global Key")
    println (s"findKey () = ${db.findKey ()}")

    banner ("4NF Decomposition?")
    db._4nf_decomp (mvd)
    db._4nf_root.printTree ()

end normalizationTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `normalizationTest4` main method tests the `Normalization` class.
 *  This example tests the BCNF Decomposition Algorithm and 3NF Synthesis Algorithm.
 *  U = Course, Teacher, Student, Grade, Hour, Room
 *
 *  fd = C  -> T
 *       CS -> G
 *       HR -> C
 *       HS -> R
 *       HT -> R
 *
 *  > runMain scalation.database.normalizationTest4
 */
@main def normalizationTest4 (): Unit =

    val r  = SET ('C', 'T', 'H', 'R', 'S', 'G')
    val fd = VEC (FD (SET ('C'), SET ('T')),
                  FD (SET ('C', 'S'), SET ('G')),
                  FD (SET ('H', 'R'), SET ('C')),
                  FD (SET ('H', 'S'), SET ('R')),
                  FD (SET ('H', 'T'), SET ('R')))

    val db = Normalization (r, fd)

    banner ("BCNF Decomposition Algorithm")
    db.bcnf_decomp ()
    db.bcnf_root.printTree ()

    val p = db.bcnf_root.collectLeaves ()

    banner ("Dependency Preservation?")
    for f <- fd do
        println (s"f = $f, preserve (f, p) = ${db.preserve (f, p)}")

    banner ("3NF Synthesis Algorithm")
    val pp = db._3nf_synthesis ()
    for ri <- pp do println (ri)

    banner ("Find a Global Key")
    println (s"findKey () = ${db.findKey ()}")

end normalizationTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `normalizationTest5` main method tests the `Normalization` class.
 *  This example tests the 3NF Synthesis Algorithm.
 *
 *  fd = C -> ST          Cname  -> Street cciTy
 *       A -> BCN         Accno  -> Bname Cname balaNce
 *       B -> EY          Bname  -> assEts bcitY
 *       L -> BCM         Loanno -> Bname Cname aMount
 *
 *  > runMain scalation.database.normalizationTest5
 */
@main def normalizationTest5 (): Unit =

    val r  = SET ('A', 'B', 'C', 'E', 'L', 'M', 'N', 'S', 'T', 'Y')
    val fd = VEC (FD (SET ('C'), SET ('S')),                       // MINIMAL COVER
                  FD (SET ('C'), SET ('T')),
                  FD (SET ('A'), SET ('B')),
                  FD (SET ('A'), SET ('C')),
                  FD (SET ('A'), SET ('N')),
                  FD (SET ('B'), SET ('E')),
                  FD (SET ('B'), SET ('Y')),
                  FD (SET ('L'), SET ('B')),
                  FD (SET ('L'), SET ('C')),
                  FD (SET ('L'), SET ('M')))

    val db = Normalization (r, fd)

    banner ("3NF Synthesis Algorithm")
    val pp = db._3nf_synthesis ()
    for ri <- pp do println (ri)

    banner ("Merge FDs with common LHS")
    val mfd = db.merge_fds ()
    println (s"mfd = $mfd")

    banner ("BCNF Decomposition Algorithm")
    db.bcnf_decomp (fd_ = mfd)
    db.bcnf_root.printTree ()

    banner ("Find a Global Key")
    println (s"findKey () = ${db.findKey ()}")

end normalizationTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `normalizationTest6` main method tests the `Normalization` class.
 *  This example tests the Minimal Cover Algorithm.
 *
 *  fd = C -> ST          Cname  -> Street cciTy
 *       A -> BCN         Accno  -> Bname Cname balaNce
 *       B -> EY          Bname  -> assEts bcitY
 *       L -> BCM         Loanno -> bname cname aMount
 *
 *  > runMain scalation.database.normalizationTest6
 */
@main def normalizationTest6 (): Unit =

    val r  = SET ('A', 'B', 'C', 'E', 'L', 'M', 'N', 'S', 'T', 'Y')
    val fd = VEC (FD (SET ('C'), SET ('S', 'T')),
                  FD (SET ('A'), SET ('B', 'C', 'N')),
                  FD (SET ('B'), SET ('E', 'Y')),
                  FD (SET ('L'), SET ('B', 'C', 'M')),
                  FD (SET ('A', 'L'), SET ('N')))

    val db = Normalization (r, fd)

    banner ("Create Minimal Cover")
    println (s"Minimal Cover = ${db.minimal_cover ()}")

    banner ("To Check Merge FDs with common LHS")
    val mfd = db.merge_fds ()
    println (s"mfd = $mfd")

    banner ("Find a Global Key")
    println (s"findKey () = ${db.findKey ()}")

end normalizationTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `normalizationTest7` main method tests the `Normalization` class.
 *  This example tests the Dependency Preservation of db design p.
 *
 *  fd = A -> B
 *       B -> C
 *       C -> D
 *       D -> A
 *  p = { AB, BC, CD }
 *
 *  > runMain scalation.database.normalizationTest7
 */
@main def normalizationTest7 (): Unit =

    val r  = SET ('A', 'B', 'C', 'D')
    val fd = VEC (FD (SET ('A'), SET ('B')),
                  FD (SET ('B'), SET ('C')),
                  FD (SET ('C'), SET ('D')),
                  FD (SET ('D'), SET ('A')))
    val p  = VEC (SET ('A', 'B'),
                  SET ('B', 'C'),
                  SET ('C', 'D'))

    val db = Normalization (r, fd)

    for f <- fd do
        banner (s"Is FD $f Preserved?")
        println (s" preserved = ${db.preserve (f, p)}")

    banner ("Find a Global Key")
    println (s"findKey () = ${db.findKey ()}")

end normalizationTest7


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `normalizationTest8` main method tests the `Normalization` class.
 *  This example tests the Dependency Preservation of db design p.
 *  Compare with Test6.
 *
 *  fd = A -> B                                                    // same FD as Test6
 *       B -> C                                                    // same FD as Test6
 *       D -> C                                                    // LHS & RHS swapped for this FD
 *       A -> D                                                    // LHS & RHS swapped for this FD
 *  p = { AB, BC, CD }
 *
 *  > runMain scalation.database.normalizationTest8
 */
@main def normalizationTest8 (): Unit =

    val r  = SET ('A', 'B', 'C', 'D')
    val fd = VEC (FD (SET ('A'), SET ('B')),
                  FD (SET ('B'), SET ('C')),
                  FD (SET ('D'), SET ('C')),
                  FD (SET ('A'), SET ('D')))
    val p  = VEC (SET ('A', 'B'),
                  SET ('B', 'C'),
                  SET ('C', 'D'))

    val db = Normalization (r, fd)

    for f <- fd do
        banner (s"Is FD $f Preserved?")
        println (s" preserved = ${db.preserve (f, p)}")

    banner ("Find a Global Key")
    println (s"findKey () = ${db.findKey ()}")

end normalizationTest8


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `normalizationTest9` main method tests the `Normalization` class.
 *  This example tests the findKey method (see Example 2 from FindKey paper).
 *
 *  fd = AD -> B
 *       AB -> E
 *       C  -> D
 *       B  -> C
 *       AC -> F
 *
 *  > runMain scalation.database.normalizationTest9
 */
@main def normalizationTest9 (): Unit =

    val r  = SET ('A', 'B', 'C', 'D', 'E', 'F')                    // key = AB
//  val r  = SET ('A', 'B', 'C', 'D', 'E', 'F', 'G')               // key = ABG
    val fd = VEC (FD (SET ('A', 'D'), SET ('B')),
                  FD (SET ('A', 'B'), SET ('E')),
                  FD (SET ('C'), SET ('D')),
                  FD (SET ('B'), SET ('C')),
                  FD (SET ('A', 'C'), SET ('F')))

    for f <- fd do println (f)

    val db = Normalization (r, fd)

    banner ("Find a Global Key")
    println (s"findKey () = ${db.findKey ()}")

end normalizationTest9

