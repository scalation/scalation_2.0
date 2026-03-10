
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Jun 17 11:19:14 EDT 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Base Trait for Row-Oriented and Column-Oriented Relational DBMSs
 *
 *  RA Operators: rename, project, selproject, select, union, minus, intersect, product,
 *                join, leftJoin, rightJoin, divide, groupBy, aggregate, orderBy
 *
 *  Most of the RA Operators have Unicode versions: @see `scalation.UnicodeTest`
 *  @see http://milde.users.sourceforge.net/LUCR/Math/unimathsymbols.pdf
 *  
 *  inline def ρ (newName: String): T = rename (newName)
 *  inline def π (x: String): T = project (strim (x))
 *  inline def π (cPos: IndexedSeq [Int]): T = project (cPos)
 *  inline def π_ (x: String): T = project (schema diff strim (x))
 *  inline def σπ (a: String, apred: APredicate): T = selproject (a, apred)
 *  inline def σ (a: String, apred: APredicate): T = select (a, apred)
 *  inline def σ (predicate: Predicate): T = select (predicate)
 *  inline def σ (condition: String): T = select (condition)
 *  inline def σ (pkey: KeyType): T = select (pkey)
 *  inline def ⋃ (r2: T): T = union (r2)
 *  inline def - (r2: T): T = minus (r2)
 *  inline def ⋂ (r2: T): T = intersect (r2)
 *  inline def × (r2: T): T = product (r2)
 *  inline def ⋈ (predicate: Predicate2, r2: T): T = join (predicate, r2)
 *  inline def ⋈ (condition: String, r2: T): T = join (condition, r2)
 *  inline def ⋈ (x: String, y: String, r2: T): T = join (strim (x), strim (y), r2)
 *  inline def ⋈ (fkey: (String, T)): T = join (fkey)
 *  inline def ⋈ (r2: T): T = join (r2)
 *  inline def ⋉ (x: Schema, y: Schema, r2: T): T = leftJoin (x, y, r2)
 *  inline def ⋊ (x: Schema, y: Schema, r2: T): T = rightJoin (x, y, r2)
 *  inline def / (r2: T): T = divide (r2)
 *  inline def ÷ (r2: T): T = divide (r2)
 *  inline def γ (ag: String): T = groupBy (ag)
 *  inline def ℱ (ag: String, f_as: (AggFunction, String)*): T = aggregate (ag, f_as :_*)
 *  inline def ↑ (x: String*): T = orderBy (x :_*)
 *  inline def ↓ (x: String*): T = orderByDesc (x :_*)
 *
 *  Join Algorithms:
 *  r join s       NESTED LOOP JOIN (NLJ)
 *  r join_ s      INDEX JOIN (IJ) -- uses UNIQUE INDEX on the right (primary key) table (s)
 *                 may use r join s for this when case is unambiguous
 *  r _join s      INDEX JOIN (IJ) -- uses NON-UNIQUE INDEX on the left (foreign key) table (r)
 *  r _join_ s     SORT-MERGE JOIN (SMJ)
 */

package scalation
package database

import scala.collection.mutable.{ArrayBuffer => Bag, IndexedSeq, Map}
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat.{MatrixD, VectorD, VectorI, VectorL, VectorS, VectorT}

/** The base directory for data files
 */
val BASE_DIR = DATA_DIR + "database" + ⁄

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Type definitions for components of the relational model.
 *  @see `ValueType` in scalation package
 */
type Schema      = Array [String]
type Domain      = Array [Char]
type Tuple       = Array [ValueType]
type Predicate   = Tuple => Boolean
type Predicate2  = (Tuple, Tuple) => Boolean
type AggFunction = Array [ValueType] => ValueType

type APredicate  = ValueType => Boolean
type APredicate2 = (String, String, (ValueType, ValueType) => Boolean)
type Row         = Vector [ValueType]

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return whether schema x is a subset-of schema y.
 *  @param x  the first schema (array/set of attributes)
 *  @param y  the schema schema (array/set of attributes)
 */
inline def subset (x: Schema, y: Schema): Boolean = x.forall (y contains _)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Efficient alternative to Scala's 'intersect' method.
 *  @see `StrictOptimizedSeqOps`
 *  @param x  the first schema (array/set of attributes)
 *  @param y  the schema schema (array/set of attributes)
 */
inline def meet (x: Schema, y: Schema): Schema = (for a <- x if y contains a yield a).toArray

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Definitions of comparison operators for `ValueType`.
 */
inline def == (x: ValueType, y: ValueType): Boolean = x == y
inline def != (x: ValueType, y: ValueType): Boolean = x != y
inline def ne (x: ValueType, y: ValueType): Boolean = x != y
inline def <  (x: ValueType, y: ValueType): Boolean = x < y
inline def <= (x: ValueType, y: ValueType): Boolean = x <= y
inline def >  (x: ValueType, y: ValueType): Boolean = x > y
inline def >= (x: ValueType, y: ValueType): Boolean = x >= y

inline def equ (x: ValueType, y: ValueType): Boolean = x == y
inline def neq (x: ValueType, y: ValueType): Boolean = x != y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Split and trim the comma-separated names contained in the given string str.
 *  @param str  the string to split and trim
 *  @param sep  the separation character
 */
def strim (str: String, sep: Char = ','): Array [String] =
    if str == null then null
    else (str split sep).map (_.trim)
end strim

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Split a condition string into three tokens: "atr op value".
 *  e.g., "cname == 'John Doe'", "accno == 123", "balance > 1000.00", "cname == cname"
 *  @param condition  the simple condition string to parse
 */
def parseCond (condition: String): (Array [String], Boolean) =
    val token = strim (condition, '\'')
    if token.size > 1 then
        val prefix = token(0)
        val part   = strim (prefix, ' ')
        (Array (part(0), part(1), token(1)), false)
    else
        val part = strim (condition, ' ')
        println (s"part = ${stringOf (part)}")
        val twoAtrs = java.lang.Character.isUnicodeIdentifierStart (part(2)(0))
        (part, twoAtrs)
end parseCond

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Combine two schemas and disambiguate any repeated names by appending "2".
 *  @param sch1  the first schema
 *  @param sch2  the second schema
 */
def disambiguate (sch1: Schema, sch2: Schema): Schema =
    val sch = Bag.from (sch1)
    for s <- sch2 do sch += (if sch1 contains s then s + "2" else s)
    sch.toArray
end disambiguate


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Tabular` trait defines relational algebra operators.
 *  Supported domains/data-types are 'D'ouble, 'I'nt, 'L'ong, 'S'tring, and 'T'imeNum.
 *      'D' - `Double`   - `VectorD` -  64 bit double precision floating point number
 *      'I' - `Int`      - `VectorI` -  32 bit integer
 *      'L' - `Long`     - `VectorL` -  64 bit long integer
 *      'S' - `String`   - `VectorS` -  variable length numeric string
 *      'X' - `String`   - `VectorS` -  variable length numeric string (for 2x column width)
 *      'T' - `TimeNum`  - `VectorT` -  time numbers for date-time
 *  Note:  Uses F-Bounded Polymorphism so extending classes don't need to cast
 *  @param name    the name of the table
 *  @param schema  the attributes for the table
 *  @param domain  the domains/data-types for the attributes ('D', 'I', 'L', 'S', 'X', 'T')
 *  @param key     the attributes forming the primary key
 */
trait Tabular [T <: Tabular [T]] (val name: String, val schema: Schema, val domain: Domain, val key: Schema)
      extends Serializable:

    private val flaw = flawf ("Tabular")                                      // flaw function

    val on = Map [String, Int] ()                                             // map from attribute name to column number
    for j <- schema.indices do on += schema(j) -> j

    if schema.size != domain.size then flaw ("init", "size mismatch between attributes and domains")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the size in terms of number of rows in the table.
     */
    def rows: Int

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the size in terms of number of columns in the table.
     */
    inline def cols: Int = schema.size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the cardinality (number of tuples) and arity (number of attributes).
     */
    inline def dims: (Int, Int) = (rows, cols)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the range of row numbers for the table.
     */
    inline def indices: Range = 0 until rows

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the j-th column in this table (or the passed in tuples) as an array of value-type.
     *  @param j     the column to return
     *  @param tups  the collection of tuples to use (defaults to all tuples in this table)
     */
//  def col (j: Int, tups: Bag [Tuple]): Array [ValueType]

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the range of columns numbers for the table.
     */
    inline def colIndices: Range = 0 until schema.size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this table contains tuple u.
     *  @param u  the tuple to look for
     */
    def contains (u: Tuple): Boolean

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check the size of the tuple (number of elements) as well as the type of each
     *  value to ensure it is from the right domain (satisfies the DOMAIN CONSTRAINTS).
     *  @param t  the tuple to be type checked
     */
    def typeCheck (t: Tuple): Boolean =
        if t.size != domain.size then
            flaw ("typeCheck", s"$name: the size of tuple ${stringOf (t)} != ${domain.size} (the domain size)")
            return false
        var matches = true
        var j = 0
        cfor (matches && j < t.size, j += 1) {
            val type_t_j = typeOf (t(j)).head
            val dom_j    = if domain(j) == 'X' then 'S' else domain(j)
            if type_t_j != dom_j then
                flaw ("typeCheck", s"$name: domain constraint violation: tuple ${stringOf (t)} has wrong type for $j-th domain")
                matches = false
        } // cfor
        true
    end typeCheck

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add LINKAGE from this table to the refTab, by adding a FOREIGN KEY CONSTRAINT
     *  to this table specifying the foreign key attribute fkey and the table it
     *  references refTab.
     *  Caveat:  a foreign key may not be composite.
     *  @param fkey    the foreign key attribute
     *  @param refTab  the table being referenced (to its primary key)
     */
    def addLinkage (fkey: String, refTab: T): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check that all the foreign keys values in tuple t satisfy their
     *  REFERENTIAL INTEGRITY CONSTRAINTS.
     *  @param t  the tuple being checked for referential integrity
     */
    def referenceCheck (t: Tuple): Boolean

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the i-th primary key.
     *  @param i  the index in the tuples/row index
     */
    def getPkey (i: Int): KeyType

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** CREATE/recreate the primary INDEX that maps the primary key to the tuple
     *  containing it.  Warning, creating an index will remove DUPLICATES based
     *  on maintaining UNIQUENESS CONSTRAINT of primary key values.
     *  @param rebuild  if rebuild is true, use old index to build new index; otherwise, create new index
     */
    def create_index (rebuild: Boolean = false): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** DROP the primary INDEX that maps the primary key to the tuple containing it.
     */
    def drop_index (): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the table restricted to the given range of rows.
     *  @param rng  the given range of rows
     */
    def apply (rng: Range): T

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the table restricted to the given collection of rows.
     *  @param pos  the given collection of rows
     */
    def apply (pos: collection.immutable.IndexedSeq [Int]): T

    // ================================================================== RENAME

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** RENAME this table, returning a shallow copy of this table.
     *  @param newName  the new name for the table.
     */
    def rename (newName: String): T

    inline def ρ (newName: String): T = rename (newName)

    // ================================================================= PROJECT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** PROJECT the tuples in this table onto the given attribute names.
     *  @param x  the schema/attribute names to project onto
     */
    def project (x: Schema): T

    inline def project (x: String): T = project (strim (x))

    inline def π (x: String): T = project (strim (x))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** PROJECT onto the columns with the given column positions.
     *  @param cPos  the column positions to project onto
     */
    def project (cPos: IndexedSeq [Int]): T

    inline def π (cPos: IndexedSeq [Int]): T = project (cPos)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** PROJECTOUT the tuples in this table onto 'schema - the given attribute names'.
     *  @param x  the schema/attribute names to project out
     */
    inline def projectOut (x: Schema): T = project (schema diff x)

    inline def projectOut (x: String): T = project (schema diff strim (x))

    inline def π_ (x: String): T = project (schema diff strim (x))

    // ========================================================== SELECT-PROJECT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT elements from column a in this table that satisfy the atomic
     *  predicate apred and project onto that column.
     *  @param a      the attribute name of the column used for selection
     *  @param apred  the atomic predicate (`Boolean` function) to be satisfied
     */
    def selproject (a: String, apred: APredicate): T

    inline def σπ (a: String, apred: APredicate): T = selproject (a, apred)

    // ================================================================== SELECT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT the tuples in this table that satisfy the atomic predicate on column a.
     *  @param a      the attribute name of the column used for selection
     *  @param apred  the atomic predicate (`Boolean` function) to be satisfied
     */
    def select (a: String, apred: APredicate): T

    inline def σ (a: String, apred: APredicate): T = select (a, apred)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT the tuples in this table that satisfy the predicate.
     *  @param predicate  the predicate (`Boolean` function) to be satisfied
     */
    def select (predicate: Predicate): T

    inline def σ (predicate: Predicate): T = select (predicate)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT the tuples in this table that satisfy the given simple (3 token) condition.
     *  @param condition  the simple condition string "a1 op a2" to be satisfied, where
     *                    a1 is attribute, op is comparison operator, a2 is attribute or value
     */
    def select (condition: String): T

    inline def σ (condition: String): T = select (condition)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SELECT via the INDEX the tuple with the given primary key value pkey.
     *  Returns an empty table if the primary index has not been created.
     *  @param pkey  the primary key value
     */
    def select (pkey: KeyType): T

    inline def σ (pkey: KeyType): T = select (pkey)

    // =========================================================== SET OPERATORS

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** UNION this table and r2.  Check that the two tables are compatible.
     *  If they are not, return the first table.
     *  Caveat:  Assumes the key from the first table still works (@see create_index)
     *  Acts like union-all, so to remove duplicates call create_index after union.
     *  @param r2  the second table
     */
    infix def union (r2: T): T

    inline def ⋃ (r2: T): T = union (r2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute this table MINUS (set difference) table r2 (this - r2).  Check that
     *  the two tables are compatible.  If they are not, return the first table.
     *  @param r2  the second table
     */
    infix def minus (r2: T): T

    inline def - (r2: T): T = minus (r2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** INTERSECT this table and r2.  Check that the two tables are compatible.
     *  If they are not, return the first table.
     *  @param r2  the second table
     */
    infix def intersect (r2: T): T

    inline def ⋂ (r2: T): T = intersect (r2)

    // ================================================================= PRODUCT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the CARTESIAN PRODUCT of this table and r2 (this × r2).
     *  @param r2  the second table
     */
    infix def product (r2: T): T

    inline def × (r2: T): T = product (r2)

    // ==================================================================== JOIN

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** JOIN this table and r2 keeping concatenated tuples that satisfy the predicate.
     *  Caveat:  Assumes both keys are needed for the new key (depending on the
     *           predicate both may not be required).
     *  @param predicate  the join predicate to be satisfied
     *  @param r2         the second table
     */
    def join (predicate: Predicate2, r2: T): T

    inline def ⋈ (predicate: Predicate2, r2: T): T = join (predicate, r2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the THETA-JOIN of this table and r2 keeping concatenated tuples that
     *  satisfy the given simple (3 token) condition.
     *  @param condition  the simple condition "a1 op a2"
     *  @param r2         the second table
     */
    def join (condition: String, r2: T): T

    inline def ⋈ (condition: String, r2: T): T = join (condition, r2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the EQUI-JOIN via the NESTED LOOP JOIN (NLJ) algorithm of this table
     *  and r2 keeping concatenated tuples that are equal on specified attributes.
     *  @param x   the subschema/attributes for the first/this table
     *  @param y   the subschema/attributes for the second table
     *  @param r2  the second table
     */
    def join (x: Schema, y: Schema, r2: T): T

    inline def join (x: String, y: String, r2: T): T =
        join (strim (x), strim (y), r2)
    end join

    inline def ⋈ (x: String, y: String, r2: T): T = join (strim (x), strim (y), r2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the EQUI-JOIN via the INDEX JOIN (IJ) algorithm of this table and the
     *  referenced table keeping concatenated tuples that are equal on the primary key
     *  and foreign key attributes.  Uses a UNIQUE INDEX (UI) on the primary key.
     *  Caveat:  Requires the foreign key table to be first [ fkey_table join ((fkey, pkey_table) ].
     *  Usage:   deposit join (("cname", customer))
     *           as if join_
     *  @param ref  the foreign key reference (foreign key attribute, referenced table)
     */
    def join (ref: (String, T)): T

    inline def ⋈ (fkey: (String, T)): T = join (fkey)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the EQUI-JOIN via the INDEX JOIN (IJ) algorithm of this table and the
     *  referenced table keeping concatenated tuples that are equal on the primary key
     *  and foreign key attributes.  Uses a NON-UNIQUE INDEX (NUI) on the foreign key.
     *  Caveat:  Requires the foreign key table to be first [ fkey_table _join ((fkey, pkey_table) ].
     *  Usage:   deposit _join (("cname", customer))
     *  @param ref  the foreign key reference (foreign key attribute, referenced table)
     */
    def _join (ref: (String, T)): T

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the EQUI-JOIN via the SORT-MERGE JOIN (SMJ) algorithm of this table and the
     *  referenced table keeping concatenated tuples that are equal on the primary key
     *  and foreign key attributes.  Requires both tables to be ordered.
     *  Caveat:  Requires the foreign key table to be first [ fkey_table _join ((fkey, pkey_table) ].
     *  Usage:   deposit _join (("cname", customer))
     *  @param ref  the foreign key reference (foreign key attribute, referenced table)
     */
    def _join_ (ref: (String, T)): T

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the NATURAL JOIN via the NESTED LOOP JOIN (NLJ) algorithm of this table and
     *  r2 keeping concatenated tuples that agree on the common attributes.
     *  Usage:   deposit join customer
     *  @param r2  the second table
     */
    infix def join (r2: T): T

    inline def ⋈ (r2: T): T = join (r2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the NATURAL JOIN via the INDEX JOIN (IJ) algorithm of this table and
     *  r2 keeping concatenated tuples that agree on the common attributes.  Uses a
     *  UNIQUE INDEX on the primary key.
     *  Caveat:  Requires the foreign key table to be first [ fkey_table join_ ((fkey, pkey_table) ].
     *  Usage:   deposit join_ customer
     *  @param r2  the second table
     */
    infix def join_ (r2: T): T

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the NATURAL JOIN via the INDEX JOIN (IJ) algorithm of this table and
     *  r2 keeping concatenated tuples that agree on the common attributes.  Uses a
     *  NON-UNIQUE INDEX on the foreign key.
     *  Caveat:  Requires the foreign key table to be first [ fkey_table _join ((fkey, pkey_table) ].
     *  Usage:   deposit _join customer
     *  @param r2  the second table
     */
    infix def _join (r2: T): T

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the NATURAL JOIN via the SORT-MERGE JOIN (SMJ) algorithm of this table and
     *  r2 keeping concatenated tuples that agree on the common attributes.  Requires
     *  both tables to be ordered.
     *  Caveat:  Requires the foreign key table to be first [ fkey_table _join_ pkey_table ].
     *  Usage:   deposit _join_ customer
     *  @param r2  the second table
     */
    infix def _join_ (r2: T): T

    // ============================================================== OUTER JOIN

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the LEFT-EQUI-JOIN of this table and r2 keeping concatenated tuples
     *  that are equal on specified attributes.  Also, keep all tuples in the left
     *  table padding the missing attributes with null.
     *  @param x   the subschema/attributes for the left/first/this table
     *  @param y   the subschema/attributes for the right/second table
     *  @param r2  the second table
     */
    def leftJoin (x: Schema, y: Schema, r2: T): T

    // Note: although this is the semi-join symbol, due to Unicode limitations, it is used for left-join.

    inline def ⋉ (x: Schema, y: Schema, r2: T): T = leftJoin (x, y, r2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the LEFT-JOIN of this table and r2 keeping concatenated tuples
     *  that are equal on specified attributes.  Also, keep all tuples in the left
     *  table padding the missing attributes with null.
     *  @param r2  the second table
     */
    infix def leftJoin (r2: T): T = 
        val x = schema intersect r2.schema
        leftJoin (x, x, r2)
    end leftJoin

    inline def ⋉ (r2: T): T = leftJoin (r2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the RIGHT-EQUI-JOIN of this table and r2 keeping concatenated tuples
     *  that are equal on specified attributes.  Also, keep all tuples in the right
     *  table padding the missing attributes with null.
     *  @param x   the subschema/attributes for the left/first/this table
     *  @param y   the subschema/attributes for the right/second table
     *  @param r2  the second table
     */
    def rightJoin (x: Schema, y: Schema, r2: T): T = r2.leftJoin (y, x, this.asInstanceOf [T])

    inline def ⋊ (x: Schema, y: Schema, r2: T): T = rightJoin (x, y, r2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the RIGHT-JOIN of this table and r2 keeping concatenated tuples
     *  that are equal on specified attributes.  Also, keep all tuples in the right
     *  table padding the missing attributes with null.
     *  @param r2  the second table
     */
    def rightJoin (r2: T): T = 
        val x = schema intersect r2.schema
        rightJoin (x, x, r2)
    end rightJoin

    inline def ⋊ (r2: T): T = rightJoin (r2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a tuple with missing values for each column according to the given
     *  domains.  This method is used by leftJoin.
     *  @param domain  the domains of the table for which a null tuple is required
     */
    def nullTuple (domain: Domain): Tuple =
        val v = Array.ofDim [ValueType] (domain.size)
        for j <- v.indices do
            v(j) = domain(j) match 
            case 'D' => NO_DOUBLE
            case 'I' => NO_INT
            case 'L' => NO_LONG
            case 'S' => NO_STRING
            case 'T' => NO_TIMENUM
            case _   => { flaw ("nullTuple", s"does not support domain type ${domain(j)}"); null }
        end for
        v
    end nullTuple

    // ================================================================== DIVIDE

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** DIVIDE this table by table r2.  Requires a tuple in the quotient part of
     *  this table to be paired with all tuples in table r2.
     *  @param r2  the second table
     */  
    def divide (r2: T): T

    inline def / (r2: T): T = divide (r2)

    inline def ÷ (r2: T): T = divide (r2)

    // ================================================================ GROUP BY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** GROUP this table BY the specified attribute, returning this table.
     *  Each value for attribute ag will be mapped to a collection of tuples.
     *  @param ag  the attribute to group by
     */
    def groupBy (ag: String): T

    inline def γ (ag: String): T = groupBy (ag)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Assuming this table has been grouped by attribute ag, create a table 
     *  where the first column is ag and the rest are AGGREGATE FUNCTIONs applied
     *  to their corresponding attributes.
     *  @param ag    the attribute the table has been grouped on
     *  @param f_as  the aggregate function and the attribute to apply it to (as varargs)
     */
    def aggregate (ag: String, f_as: (AggFunction, String)*): T

    inline def ℱ (ag: String, f_as: (AggFunction, String)*): T = aggregate (ag, f_as*)

    // ================================================================ ORDER BY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** ORDER-BY the given attributes, i.e., reorder the tuples in this table into
     *  'ascending' order.  A stable sorting is used to allow sorting on multiple attributes.
     *  @param x  the subschema/attributes to order by
     */
    def orderBy (x: String*): T

    inline def ↑ (x: String*): T = orderBy (x*)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** ORDER-BY-DESC the given attributes, i.e., reorder the tuples in this table into
     *  'descending' order.  A stable sorting is used to allow sorting on multiple attributes.
     *  @param x  the subschema/attributes to order by
     */
    def orderByDesc (x: String*): T

    inline def ↓ (x: String*): T = orderByDesc (x*)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the basic statistics for each column of this table.
     */
    def stats: T

    // ================================================================= UPDATES

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** ADD (insert) tuple t into this table checking to make sure the domains are correct.
     *  Also, checks referential integrity for any foreign keys in the tuple.
     *  Return true iff the tuple passes the type check and reference check.
     *  @param t  the tuple to be inserted
     */
    def add (t: Tuple): T

    inline def add (v: ValueType*): T = add (v.toArray)

    inline def += (v: ValueType*): T = add (v.toArray)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** UPDATE the column with attribute name a using newVal for elements with value
     *  matchVal.  Return true iff at least one tuple is updated.
     *  @param a         the attribute name for the column to be updated
     *  @param newVal    the value used to assign updated values
     *  @param matchVal  the value to be matched to elements
     */
    def update (a: String, newVal: ValueType, matchVal: ValueType): Boolean

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** UPDATE the column with attribute name a using function func for elements with
     *  value matchVal.  Return true iff at least one tuple is updated.
     *  @param a         the attribute name for the column to be updated
     *  @param func      the function used to assign updated values
     *  @param matchVal  the value to be matched to elements
     */
    def update (a: String, func: ValueType => ValueType, matchVal: ValueType): Boolean

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** DELETE all tuples in this table satisfying the deletion predicate.
     *  If there is an index, remove those tuples from the index as well.
     *  Return true iff at least one tuple is deleted.
     *  @param predicate  the predicate that specifies which tuples to delete
     */
    def delete (predicate: Predicate): Boolean

    // ================================================================= CONVERT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this table to a matrix of doubles by making the necessary
     *  type transformations.
     *  @param cols  the column positions to use for forming the matrix
     */
    def toMatrix (cols: Array [Int]): MatrixD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this table to a matrix and a vector of doubles by making the necessary
     *  type transformations.
     *  Usage: table -> (X, y) in linear algebra/regression problem Xb = y.
     *  @param cols  the column positions to use for forming the matrix
     *  @param colj  the column position to use for forming the vector
     */
    def toMatrixV (cols: Array [Int] = Array.range (0, schema.size-1),
                   colj: Int = schema.size-1): (MatrixD, VectorD)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the colj column of this relation into a vector of doubles, etc.
     *  @param colj  the column position to use for the vector
     */
    def toVectorD (colj: Int = 0): VectorD
    def toVectorI (colj: Int = 0): VectorI
    def toVectorL (colj: Int = 0): VectorL
    def toVectorS (colj: Int = 0): VectorS
    def toVectorT (colj: Int = 0): VectorT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this table and r2 are incompatible by having differing domains.
     *  @param r2  the second table
     */
    def incompatible (r2: T): Boolean =
         val (dom, dom2) = (stringOf (domain), stringOf (r2.domain))
         if dom != dom2 then
             flaw ("incompatible", s"$name and ${r2.name} have differing domains $dom vs. ${dom2}")
             true
         else
             false                                                      // i.e., they are compatible
    end incompatible

    // ==================================================================== PULL

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Pull the values out of tuple t for the attributes in subschema x.
     *  @param t  the given tuple to pull values out of
     *  @param x  the subschema/attributes to be collected
     */
    inline def pull (t: Tuple, x: Schema): Tuple = (for a <- x yield t(on(a))).toArray

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Pull the values out of tuple t for the attributes in subschema column positions cp.
     *  It is more EFFICIENT to use column positions rather than attribute names.
     *  @param t   the given tuple to pull values out of
     *  @param cp  the subschema/attribute column positions to be collected
     */
    inline def pull (t: Tuple, cp: IndexedSeq [Int]): Tuple =
        val tup = Array.ofDim [ValueType] (cp.size)
        for i <- tup.indices do tup(i) = t(cp(i))
        tup
    end pull

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Pull the values out of row t for the attributes in subschema x.
     *  @param t  the given row to pull values out of
     *  @param x  the subschema/attributes to be collected
     */
    inline def pull (t: Row, x: Schema): Row = (for a <- x yield t(on(a))).toVector

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Pull the values out of row t for the attributes in subschema column positions cp.
     *  @param t   the given row to pull values out of
     *  @param cp  the subschema/attribute column positions to be collected
     */
    inline def pull (t: Row, cp: IndexedSeq [Int]): Row = (for c <- cp yield t(c)).toVector

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Pull a value out of tuple t for attribute a.
     *  @param t  the given tuple to pull value out of
     *  @param a  the attribute to be collected
     */
    inline def pull (t: Tuple, a: String): ValueType = t(on(a))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Pull the domains out of this table for the attributes in subschema x.
     *  @param x  the subschema/attributes to be collected
     */
    inline def pull (x: Schema): Domain = (for a <- x yield domain(on(a))).toArray

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Pull the domains out of this table for the attributes in subschema column positions cp.
     *  @param cp  the subschema/attribute column positions to be collected
     */
    inline def pull (cp: IndexedSeq [Int]): Domain = (for c <- cp yield domain(c)).toArray

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Pull the column positions within this table for the attributes in subschema x.
     *  It is more EFFICIENT to pull all the column positions before looping through tuples.
     *  @param x  the subschema/attributes to be collected
     */
    inline def pullPos (x: Schema): Array [Int] =
        val pos = Array.ofDim [Int] (x.size)
        for i <- pos.indices do pos(i) = on(x(i))
        pos
    end pullPos

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the tuples in tups into a string, e.g., for displaying a collection
     *  of tuples.
     *  @param tups  the tuples to be converted to a string
     */
    def showT (tups: Bag [Tuple]): String =
        val sb = StringBuilder ()
        for t <- tups do
            sb.append ("( ")
            for v <- t do sb.append (s"$v ")
            sb.append ("), ")
        end for
        sb.append ("\n")
        sb.toString
    end showT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a copy of this table limited to the first n tuples/rows.
     *  @param n  the number of tuples/rows to keep
     */
    def limit (n: Int): T

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SHOW/print this table, one tuple per row.
     *  @param rng  the range of tuples to show, defaults to 0 until 1
     */
    def show (rng: Range = 0 until 1): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SHOW/print this table's primary index.
     */
    def show_index (): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SHOW/print this table's foreign keys.
     */
    def show_foreign_keys (): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** SAVE this table in a file using serialization.
     *  @see load in `Tabular` object
     */
    def save (): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Write this table into a Comma-Separated-Value (CSV) file with each tuple
     *  written to a line.
     *  @param fileName  the file name of the data file
     */
    def writeCSV (fileName: String): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Write this table into a JavaScript Object Notation (JSON) file.
     *  @param fileName  the file name of the data file
     */
    def writeJSON (fileName: String): Unit

end Tabular

 
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Tabular` companion object provides methods supporting prefix forms for
 *  unary relational algebra operators.
 *  Example: prefix form vs. postfix form. 
 *
 *      π("cname, ccity")(σ("balance < amount")(customer ⋈ deposit ⋈ loan))
 *
 *      (customer ⋈ deposit ⋈ loan).σ("balance < amount").π("cname, ccity")
 */
object Tabular:

    inline def project [T <: Tabular [T]] (x: String)(tab: Tabular [T]): T = tab.project (strim (x))

    inline def π [T <: Tabular [T]] (x: String)(tab: Tabular [T]): T = tab.project (strim (x))

    def select [T <: Tabular [T]] (condition: String)(tab: Tabular [T]): T = tab.select (condition)

    inline def σ [T <: Tabular [T]] (condition: String)(tab: Tabular [T]): T = tab.select (condition)

end Tabular

