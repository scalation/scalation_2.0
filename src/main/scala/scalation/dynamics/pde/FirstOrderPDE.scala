
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Mar 28 13:43:50 EDT 2013
 *  @see     LICENSE (MIT style license file)
 *  @see     www.ita.uni-heidelberg.de/~dullemond/lectures/.../Chapter_3.pdf
 *  @see     www2.hawaii.edu/~norbert/CompPhys/chapter14.pdf
 *
 *  @note    First Order Partial Differential Equation (PDE) Solver
 *           Explicit Finite Difference
 *           Supports (1) Lax-Wendroff and (2) MacCormack schemes
 *           Tested on the Advection Equation and Brugers' Equation
 */

package scalation
package dynamics
package pde

import scala.math.{abs, ceil, exp}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FirstOrderPDE` class is used to solve First Order Partial Differential
 *  Equations like the Advection Equation.  Let 'u(x, t)' = concentration in a fluid
 *  with velocity 'v' at position '0 <= x <= xm' and time 't' > 0.  Numerically solve the
 *
 *  Advection Equation:      u_t + v(x, t) u_x = 0
 *                           ∂u/∂t + v(x, t) ∂u/∂x = 0
 *  with initial conditions  u(x, 0) = ic(x)
 *      boundary conditions  (u(0, t), u(xm, t)) = bc
 *
 *  @param v   the velocity field function v(x, t)
 *  @param dt  delta 't'
 *  @param dx  delta 'x'
 *  @param xm  the length of the column
 *  @param ic  the initial conditions as a function of position 'x'
 *  @param bc  the boundary conditions as a 2-tuple for end-points 0 and 'xm'
 *  @param f   the flux (how much flow) function f(u, x)
 */
class FirstOrderPDE (v: (Double, Double) => Double,
                     dt: Double, dx: Double, xm: Double,
                     ic: FunctionS2S, bc: (Double, Double),
                     f: (Double, Double) => Double = null):

    private val nx = (ceil ((xm) / dx)).toInt + 1               // number of x values in grid
    private var u  = new VectorD (nx)                           // old concentration vectors
    private var uu = new VectorD (nx)                           // new concentration vectors

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve for the concentration of the column at time t, returning the vector of
     *  concentration representing the concentration profile of column over its length.
     *  This method uses an Explicit Finite Difference technique to solve the PDE.
     *  L-W is the Lax-Wendroff scheme which has second-order accuracy.
     *  @see   math.nju.edu.cn/~qzh/numPDE.pdf
     *  @param te  the time the solution is desired (t-end)
     */
    def solve (te: Double): VectorD =

        // if flux f is null, default to linear advection: f(u, x) = v(x, t) * u
        val flux = if f != null then f
                   else (u_val: Double, x_pos: Double) => v(x_pos, 0.0) * u_val
 
        val r  = dt / dx                                        // multiplier in recurrence equation
        val nt = (ceil (te / dt)).toInt + 1                     // number of t values in grid
        for i <- 0 until nx do u(i) = ic (i*dx)                 // apply initial conditions
        u(0) = bc._1                                            // apply boundary conditions, left only

        println (s"at t = 0.0 : \tu  = $u")

        for j <- 1 until nt do                                  // iterative over t = j*dt
            val t = j*dt                                        // current time t

            for i <- 1 until nx- 1 do                           // iterative over x = i*dx
                val x = i*dx                                    // current position x
                val f_plus  = flux(u(i+1), (i+1)*dx)            // calc. flux on positive side
                val f_minus = flux(u(i-1), (i-1)*dx)            // calc. flux on negative side

                // Calculate local wave speed 'a' at the center point (Jacobian)
                // For Burgers, this is just u(i). For Advection, this is v(x, t).
                val a = r * v(x, t)                             // wave speed
                                                                // explicit recurrence equation
                uu(i) = u(i) - 0.5 * r * (f_plus - f_minus) +   // -central flux difference
                        0.5 * a * r * (f_plus - 2.0 * flux(u(i), x) + f_minus)   // +artificial viscosity / 2nd-order correction

            end for

            println (s"at t = ${t}: \tuu = $uu")
            uu(0)  = bc._1                                      // apply left boundary condition
            val ut = uu; uu = u; u = ut                         // swap new and old references
        end for
        u                                                       // return the concentration vector
    end solve

//              uu(i) = u(i) - a * (u(i) - u(i-1))
//              uu(i) = .5 * ((1. - a) * u(i+1) + (1. + a) * u(i-1))
//              uu(i) = u(i) - .5 * a * u(i+1) + .5 * a * u(i-1)
//              uu(i) = u(i) - .5 * a * (u(i+1) - u(i-1)) + .5 * a*a * (u(i+1) - 2.0*u(i) + u(i-1))  // L-W

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve for the concentration of the column at time t, returning the vector of
     *  concentration representing the concentration profile of column over its length.
     *  This method uses an Explicit Finite Difference technique to solve the PDE.
     *  Uses the MacCormack scheme which has second-order accuracy.
     *  @see   math.nju.edu.cn/~qzh/numPDE.pdf
     *  @param te  the time the solution is desired (t-end)
     */
    def solve2 (te: Double): VectorD =

        // if flux f is null, default to linear advection: f(u, x) = v(x, t) * u
        val flux = if f != null then f
                   else (u_val: Double, x_pos: Double) => v(x_pos, 0.0) * u_val

        val r  = dt / dx                                        
        val nt = (ceil (te / dt)).toInt + 1                     
        val up = new VectorD(nx)                                // predicted vector storage
    
        for i <- 0 until nx do u(i) = ic(i*dx)                 
        u(0) = bc._1                                            

        for _ <- 1 until nt do                                  
        
            // --- 1. PREDICTOR STEP (Forward Difference) ---
            // use u(i) and u(i+1) to find the predicted value up(i)

            for i <- 0 until nx - 1 do
                up(i) = u(i) - r * (flux(u(i+1), (i+1)*dx) - flux(u(i), i*dx))
            up(nx-1) = u(nx-1)                                  // simple boundary handling for predictor

            // --- 2. CORRECTOR STEP (Backward Difference) ---
            // use up(i) and up(i-1) and then average with u(i)

            for i <- 1 until nx - 1 do
                val u_corr = up(i) - r * (flux(up(i), i*dx) - flux(up(i-1), (i-1)*dx))
                uu(i)      = 0.5 * (u(i) + u_corr)

            // --- 3. BOUNDARY & SWAP ---
            uu(0) = bc._1                                       
            uu(nx-1) = uu(nx-2)                                 // simple zero-gradient boundary for the right side
        
            val ut = uu; uu = u; u = ut            
        end for
        u                                                       // return the concentration vector
    end solve2

end FirstOrderPDE


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `firstOrderPDETest` main function is used to test the `FirstOrderPDE` class.
 *  Numerically solve the Advection Equation:  '∂u/∂t + v(x, t) * ∂u/∂x = 0'
 *  > runMain scalation.dynamics.pde.firstOrderPDETest
 */
@main def firstOrderPDETest (): Unit =

    val dt = 1.0                                               // delta t in sec
    val dx = 1.0                                               // delta x in cm
    val xm = 100.0                                             // length of column in cm

    def ic (x: Double): Double = if x < 30.0 then 1.0 else 0.0    // initial conditions
    val bc = (1.0, 0.0)                                        // boundary conditions - only a left bc

    def v (x: Double, t: Double): Double = 1.0 + 0.0 * (x+t)   // the velocity field

    val pde = new FirstOrderPDE (v, dt, dx, xm, ic, bc)
    val u1  = pde.solve (30.0)
    val u2  = pde.solve2 (30.0)

    banner ("Explicit Finite Difference for Advection Equation")
    println (s"solution LW = $u1")                             // solve for t = 1, 2, ... sec
    println (s"solution M  = $u2")                             // solve for t = 1, 2, ... sec
    println (s"difference  = ${u1 - u2}")

end firstOrderPDETest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `firstOrderPDETest2` main function is used to test the `FirstOrderPDE` class.
 *  Numerically solve the Advection Equation:  '∂u/∂t + v(x, t) * ∂u/∂x = 0'
 *  > runMain scalation.dynamics.pde.firstOrderPDETest2
 */
@main def firstOrderPDETest2 (): Unit =

    val dt = .2                                                // delta t in sec
    val dx = .2                                                // delta x in cm
    val xm = 3.0                                               // length/height of column in cm
    val d  = 0.5                                               // decay width
    val x0 = 0.0

    def ic (x: Double): Double =                               // initial conditions
        xm / (1.0 + exp ((x - x0) / d))                        // logistic function

    for i <- 0 to 15 do printf ("x = %4.1f, \t%6.3f\n", i*dx, ic (i*dx)) 

    val bc = (ic (0.0), 0.0)                                   // boundary conditions - only a left bc

    def v (x: Double, t: Double): Double = 1.0 + 0.0 * (x+t)   // the velocity field

    val pde = new FirstOrderPDE (v, dt, dx, xm, ic, bc)
    val u1  = pde.solve (30.0)
    val u2  = pde.solve2 (30.0)

    banner ("Explicit Finite Difference for Advection Equation")
    println (s"solution LW = $u1")                             // solve for t = 1, 2, ... sec
    println (s"solution M  = $u2")                             // solve for t = 1, 2, ... sec
    println (s"difference  = ${u1 - u2}")

end firstOrderPDETest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `firstOrderPDETest3` main function is used to test the `FirstOrderPDE` class.
 *  Numerically solve the Advection Equation:  '∂u/∂t + v(x, t) * ∂u/∂x = 0'
 *  @see www.public.asu.edu/~hhuang38/pde_slides_numerical.pdf
 *  > runMain scalation.dynamics.pde.firstOrderPDETest3
 */
@main def firstOrderPDETest3 (): Unit =

    val EPSILON = 1E-9                                         // a value close to zero

    val dt = .1                                                // delta t in sec
    val dx = .2                                                // delta x in cm
    val xm = 2.0                                               // length/height of column in cm

    def ic (x: Double): Double =                               // initial conditions
        if abs (x - .8) < EPSILON then 1.0 else 0.0

    val bc = (ic (0.0), 0.0)                                   // boundary conditions - only a left bc

    def v (x: Double, t: Double): Double = -1.0 + 0.0 * (x+t)  // the velocity field

    val pde = new FirstOrderPDE (v, dt, dx, xm, ic, bc)
    val u1  = pde.solve (30.0)
    val u2  = pde.solve2 (30.0)

    banner ("Explicit Finite Difference for Advection Equation")
    println (s"solution LW = $u1")                             // solve for t = 1, 2, ... sec
    println (s"solution M  = $u2")                             // solve for t = 1, 2, ... sec
    println (s"difference  = ${u1 - u2}")

end firstOrderPDETest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `firstOrderPDETest4` main function is used to test the `FirstOrderPDE` class.
 *  Numerically solve the Burgers' Equation:  ∂u/∂t + ∂f(u)/∂x = 0
 *  with Flux Function:                       f(u) = 0.5 u^2
 *  by the Chain Rule:                        ∂u/∂t + u ∂u/∂x = 0
 *  @see https://zingale.github.io/comp_astro_tutorial/advection_euler/burgers/burgers-methods.html
 *  > runMain scalation.dynamics.pde.firstOrderPDETest4
 */
@main def firstOrderPDETest4 (): Unit =

    import scala.annotation.unused

    val dt = 0.01                                              // time step
    val dx = 0.1                                               // spatial step
    val xm = 10.0                                              // domain length
    val te = 2.0                                               // total time to simulate

    // Initial Condition: A smooth wave u(x, 0) = 1.0 + 0.5 * sin(pi * x / 5)
    // The peak (at 1.5) will travel faster than the base (at 1.0)
    def ic (x: Double): Double = 1.0 + 0.5 * math.sin(math.Pi * x / 5.0)

    // Boundary Conditions: Left constant at the initial value of ic(0)
    val bc = (ic(0.0), 0.0)

    // Velocity field for Burgers' is simply u itself. 
    // For your class, we can define v as a function of the state.
    // Note: If your current solver expects v(x, t), pass a function that 
    // uses the current u(i). If it only takes (x, t), assume v = u.
    def v_burgers (x: Double, t: Double): Double = 
        // This is a placeholder; in a real quasilinear solver, 
        // the 'a' in your L-W code should be the value of u(i).
        1.25 // An average speed for stability estimation

    // Flux function for Burgers': f(u, x) -> f(u) = 0.5 * u^2
    def flux_burgers (u: Double, @unused x: Double): Double = 0.5 * u * u

    println ("Running Inviscid Burgers' Equation...")
    
    // Create the PDE solver
    val pde = new FirstOrderPDE (v_burgers, dt, dx, xm, ic, bc, flux_burgers)

    val u1 = pde.solve (te)
    val u2 = pde.solve2 (te)

    banner ("Explicit Finite Difference for Burgers Equation")
    println (s"Final state at te = $te:")
    println (s"solution LW = $u1")                             // solve for t = 1, 2, ... sec
    println (s"solution M  = $u2")                             // solve for t = 1, 2, ... sec
    println (s"difference  = ${u1 - u2}")

end firstOrderPDETest4

