

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Sep  9 13:07:48 EDT 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Facilitates of the Use of Unicode Symbols in ScalaTion (copy-paste)
 *
 *  @see     Unicode operator definitions:
 *           in scalation: `Complex`, `Rat`, `SortedSetExt`, `TimeNum`, `Util`, `ValueType`
 *           in mathstat: `MatrixD`, `VectorD`
 *           in database: `Tabular`
 *           in calculus: `Differential`,  `Integral`, `Poly`
 *           in modeling.forecasting: `ARIMA_diff`
 */

package scalation

import scala.runtime.ScalaRunTime.stringOf

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Unicode` object provides arrays to facilitate the use of Unicode.
 *  ScalaTion currently uses a few UTF-16 characters, see code below.  Most UTF-16
 *  characters are 2 bytes (16 bits).  Extended characters are encoded in 4 bytes.
 *  The 7-bit ASCII range is a subset of Unicode, the first 128 code points from U+0000 to U+007F
 *  ScalaTion limits characters to the range '\u0000' to 'U+2BFF'.
 *  Developers should test Unicode symbols here before trying them in the code.
 *  @see en.wikipedia.org/wiki/UTF-16
 *  @see www.tamasoft.co.jp/en/general-info/unicode.html
 *  @see en.wikipedia.org/wiki/List_of_Unicode_characters
 *  @see arxiv.org/abs/1112.1751
 */
object Unicode:

// operators

   //               Symbol Code Point      Description                                Category
   val op =  Array ( ('¬', '\u00AC'),     // Not Sign (Negation)                      Logic (Latin-1 Supplement)
                     ('±', '\u00B1'),     // Plus-Minus Sign                          Arithmetic
                     ('·', '\u00B7'),     // Middle Dot/Bullet Point                  Miscellaneous
                     ('×', '\u00D7'),     // Multiplication Sign                      Arithmetic
                     ('÷', '\u00F7'),     // Division Sign                            Arithmetic/Relational Algebra

                     ('ᵀ', '\u1D40'),     // Transpose (used as superscript T)        Linear Algebra
                     ('‖', '\u2016'),     // Double Vertical Line (Norm/Determinant)  Linear Algebra
                     ('‾', '\u203E'),     // Overscore                                Max, with _ for Min
                     ('⁻', '\u207B'),     // Superscript Minus Sign                   Max, Exponentiation
                     ('₋', '\u208B'),     // Subscript Minus Sign                     Min, Exponentiation

                     ('ℱ', '\u2131'),     // Script F for Aggregate Function          Relational Algebra
                     ('↑', '\u2191'),     // Upward Arrow                             Exponentiation/Sorting
                     ('↓', '\u2193'),     // Downward Arrow                           Sorting
                     ('↦', '\u21A6'),     // Rightwards Arrow From Bar (Maps To)      Mapping

                     ('∀', '\u2200'),     // For All (Universal Quantifier)           Logic
                     ('∂', '\u2202'),     // Partial Differential                     Calculus (use d for derivative)
                     ('∃', '\u2203'),     // There Exists (Existential Quantifier)    Logic
                     ('∄', '\u2204'),     // Not Exists (Existential Quantifier)      Logic
                     ('∆', '\u2206'),     // Difference                               Time Series
                     ('∇', '\u2207'),     // Nabla (Gradient, Curl, Divergence)       Calculus/Vector
                     ('∈', '\u2208'),     // Element Of                               Set Theory
                     ('∉', '\u2209'),     // Not Element Of                           Set Theory
                     ('∏', '\u220F'),     // N-ary Product                            N-ary Operator

                     ('∑', '\u2211'),     // N-ary Summation                          N-ary Operator
                     ('∘', '\u2218'),     // Ring Operator (Function Composition)     Function
                     ('√', '\u221A'),     // Square Root (Radical)                    Function
                     ('∛', '\u221B'),     // Cube Root                                Function
                     ('∝', '\u221D'),     // Proportional To                          Relational

                     ('∣', '\u2223'),     // Divides                                  Relational
                     ('∥', '\u2225'),     // Parallel To                              Relational
                     ('∧', '\u2227'),     // Logical AND (Conjunction)                Logic
                     ('∨', '\u2228'),     // Logical OR (Disjunction)                 Logic
                     ('∩', '\u2229'),     // Intersection                             Set Theory
                     ('∪', '\u222A'),     // Union                                    Set Theory
                     ('∫', '\u222B'),     // Integral                                 Calculus
                     ('∬', '\u222C'),     // Double Integral                          Calculus

                     ('≈', '\u2248'),     // Almost Equal To                          Relational

                     ('≠', '\u2260'),     // Not Equal To                             Relational
                     ('≡', '\u2261'),     // Identical To                             Relational
                     ('≤', '\u2264'),     // Less-Than or Equal To                    Relational
                     ('≥', '\u2265'),     // Greater-Than or Equal To                 Relational

                     ('⊂', '\u2282'),     // Subset Of                                Set Theory
                     ('⊃', '\u2283'),     // Superset Of                              Set Theory
                     ('⊄', '\u2284'),     // Not Subset Of                            Set Theory
                     ('⊅', '\u2285'),     // Not Superset Of                          Set Theory
                     ('⊆', '\u2286'),     // Subset Of Or Equal To                    Set Theory
                     ('⊇', '\u2287'),     // Superset Of Or Equal To                  Set Theory
                     ('⊈', '\u2288'),     // Not Subset Of Or Equal To                Set Theory
                     ('⊉', '\u2289'),     // Not Superset Of Or Equal To              Set Theory

                     ('⊕', '\u2295'),     // Circled Plus (Direct Sum)                Algebra/Linear Algebra
                     ('⊗', '\u2297'),     // Circled Times (Tensor Product)           Algebra/Linear Algebra
                     ('⊙', '\u2299'),     // Hadamard Product                         Linear Algebra
                     ('⊥', '\u22A5'),     // Perpendicular / Up tack                  Geometry/Logic
                     ('∙', '\u22C5'),     // Dot Product operator (Vector/Matrix)     Arithmetic/Linear Algebra
                     ('⋆', '\u22C6'),     // Star/Convolution                         Linear Algebra
                     ('⋈', '\u22C8'),     // Join                                     Relational Algebra
                     ('⋉', '\u22C9'),     // Left Join                                Relational Algebra
                     ('⋊', '\u22C9'),     // Right Join                               Relational Algebra

                     ('⟨', '\u27E8'),     // Left Angle Bracket (Inner Product)       Linear Algebra
                     ('⟩', '\u27E9'))     // Right Angle Bracket (Inner Product)      Linear Algebra

// special values

   //               Symbol Code Point     Description                                 Category
   val sv =  Array ( ('ŷ', '\u0177'),     // y-hat (estimate for y)                   Statistics
                     ('ƒ', '\u0192'),     // f with a hook                            ƒ = f' (derivative of f)
                     ('∅', '\u2205'),     // Empty Set                                Set Theory
                     ('∞', '\u221E'))     // Infinity                                 Miscellaneous

// Small Greek letters '\u03B1' to '\u03C9'

   //               Symbol Code Point     Description
   val grk = Array ( ('α', '\u03B1'),     // Greek Small Letter alpha
                     ('β', '\u03B2'),     // Greek Small Letter beta
                     ('γ', '\u03B3'),     // Greek Small Letter gamma
                     ('δ', '\u03B4'),     // Greek Small Letter delta
                     ('ε', '\u03B5'),     // Greek Small Letter epsilon
                     ('ζ', '\u03B6'),     // Greek Small Letter zeta
                     ('η', '\u03B7'),     // Greek Small Letter eta
                     ('θ', '\u03B8'),     // Greek Small Letter theta
                     ('ι', '\u03B9'),     // Greek Small Letter iota
                     ('κ', '\u03BA'),     // Greek Small Letter kappa
                     ('λ', '\u03BB'),     // Greek Small Letter lambda
                     ('μ', '\u03BC'),     // Greek Small Letter mu
                     ('ν', '\u03BD'),     // Greek Small Letter nu
                     ('ξ', '\u03BE'),     // Greek Small Letter xi
                     ('ο', '\u03BF'),     // Greek Small Letter omicron
                     ('π', '\u03C0'),     // Greek Small Letter pi
                     ('ρ', '\u03C1'),     // Greek Small Letter rho
                     ('ς', '\u03C2'),     // Greek Small Letter final sigma
                     ('σ', '\u03C3'),     // Greek Small Letter sigma
                     ('τ', '\u03C4'),     // Greek Small Letter tau
                     ('υ', '\u03C5'),     // Greek Small Letter upsilon
                     ('φ', '\u03C6'),     // Greek Small Letter phi
                     ('χ', '\u03C7'),     // Greek Small Letter chi
                     ('ψ', '\u03C8'),     // Greek Small Letter psi
                     ('ω', '\u03C9'))     // Greek Small Letter omega

// superscripts

    /** Unicode characters for superscripts to 0, 1, ... 9
     */
    val supc = Array ('⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹')

    /** Unicode numbers for superscripts to 0, 1, ... 9
     */
    val supn = Array ('\u2070', '\u00b9', '\u00b2', '\u00b3', '\u2074', '\u2075', '\u2076', '\u2077', '\u2078', '\u2079')

// subscripts

    /** Unicode characters for subscripts to 0, 1, ... 9
     */
    val subc = Array ('₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉')

    /** Unicode numbers for subscripts to 0, 1, ... 9
     */
    val subn = Array ('\u2080', '\u2081', '\u2082', '\u2083', '\u2084', '\u2085', '\u2086', '\u2087', '\u2088', '\u2089')

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Unicode characters for superscripts derived for integer 'i'.
     *  @param i  the integer to convert into a subscript
     */
    def sup (i: Int) = for c <- i.toString yield supc(c - '0')

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Unicode characters for subscripts derived from integer 'i'
     *  @param i  the integer to convert into a subscript
     */
    def sub (i: Int) = for c <- i.toString yield subc(c - '0')

end Unicode


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `unicodeTest` main function is used to the `Unicode` object.
 *  Show operators, special characters, and Greek letters.
 *  > runMain scalation.unicodeTest
 */
@main def unicodeTest (): Unit =

    import Unicode._

    println ("List of Unicode Operator Characters used in ScalaTion:")
    for o <- op do println (o)
 
    println ("List of Unicode Special Characters used in ScalaTion:")
    for s <- sv do println (s)

    println ("List of Unicode Greek Letters used in ScalaTion:")
    for g <- grk do println (g)

end unicodeTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `unicodeTest2` main function is used to the `Unicode` object.
 *  Show subscripts and superscripts.
 *  > runMain scalation.unicodeTest2
 */
@main def unicodeTest2 (): Unit =

    import Unicode._

    val uc = Array ('Ɛ', 'Ʋ', 'π', 'σ', '∙', '≠', '≤', '≥', '⋂', '⋃', '⋈')
    val un = Array ('\u0190', '\u01b2', '\u03c0', '\u03c3', '\u2219',
                    '\u2260', '\u2264', '\u2265', '\u22c2', '\u22c3', '\u22c8')

    println ("Unicode character:")
    for c <- uc do println (s"c = $c = U+%04x".format (c.toInt))
    println ("Unicode number:")
    for n <- un do println (s"n = $n = U+%04x".format (n.toInt))

    def ∙ () = "∙ worked"
 
    println (∙())

    println ("supc = " + stringOf (supc))
    println ("supn = " + stringOf (supn))
    println ("subc = " + stringOf (subc))
    println ("subn = " + stringOf (subn))

    println ("sup (12) = " + sup (12))
    println ("sub (12) = " + sub (12))

end unicodeTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `unicodeTest3` main function is used to the `Unicode` object.
 *  Show Scala 3 operators.
 *  First character determines precedence: The first character of an operator is the
 *  primary factor in determining its precedence.
 *  Letters vs. symbols: Operators that start with letters have the lowest precedence,
 *  while those with symbols like *, /, and % have higher precedence.
 *  @see docs.scala-lang.org/tour/operators.html
 *  @see www.geeksforgeeks.org/scala/operators-precedence-in-scala/
 *  > runMain scalation.unicodeTest3
 */
@main def unicodeTest3 (): Unit =

    println ("""
    Operator precedence table (from highest to lowest precedence) 
    Category 	Operators		Associativity
    Other Symbols  Any symbol not listed Left to right
    Postfix	() []			Left to right
    Unary	! ~			Right to left
    Multiplicative  * / %		Left to right
    Additive	+ -			Left to right
    List Cons	::                      Right to left
    Shift	>> >>> <<		Left to right
    Relational	> >= < <=		Left to right
    Equality	== !=			Left to right
    Bitwise AND	&			Left to right
    Bitwise XOR	^			Left to right
    Bitwise OR	|	        	Left to right
    Logical AND	&&			Left to Right
    Logical OR	||			Left to Right
    Assignment	= += -= *= /= %= 	Right to left
    (cont.)	>>= <<= &= ^= |=        Right to left
    Infix letters  and, or, if, etc.	Left to right
    """)

    println ("""
    Operator precedence based on first character
    (characters not shown below)
    * / %
    + -
    :
    < >
    = !
    &
    ^
    |
    (all letters, $, _)
    """)

end unicodeTest3

