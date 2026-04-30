
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simplePerceptron2` main function illustrates the use of Incremental Gradient Descent (IGD)
 *  to optimize the weights/parameters of a simple neural network (Perceptron).
 *
 *      grad g = -x.ᵀ * (y - ŷ) * ƒ    where pred ŷ = f(x * b)
 *
 *  Computations done at the scalar level: X -> y.  R^2 = .886
 *  > runMain scalation.modeling.simplePerceptron2
 */
@main def simplePerceptron2 (): Unit =

    val sst = (y - y.mean).normSq                               // sum of squares total

    val b = VectorD (0.1, 0.2, 0.1)                             // initial weights/parameters (random in practice)

    val η = 2.5                                                 // learning rate (to be tuned)
    var u, ŷ, ε, ƒ, δ: Double = NO_DOUBLE
    var g: VectorD = null
    val yp = new VectorD (y.dim)                                // save each prediction in yp

    for epoch <- 1 to 10 do
        println (s"Improvement step $epoch")
        var sse = 0.0
        for i <- x.indices do
            val (xi, yi) = (x(i), y(i))                         // randomize i for Pure Stochastic Gradient Descent (PSGD)

            // forward prop: input -> output
            u =                                                 // pre-activation scalar via dot (∙) product
            ŷ =                                                 // prediction scalar
            ε =                                                 // error scalar

            // backward prop: output -> input
            ƒ =                                                 // derivative (f') for sigmoid
            δ =                                                 // delta correction scalar
            g =                                                 // gradient vector

            // parameter update
            b -=                                                // update parameter vector

            yp(i) = ŷ                                           // save i-th prediction
            sse  += ε * ε                                       // sum of squared errors
        end for
        val r2 = 1.0 - sse / sst                                // R^2

        println (s"""
        u   = $u
        ŷ   = $ŷ
        ε   = $ε
        ƒ   = $ƒ
        δ   = $δ
        g   = $g
        b   = $b
        sse = $sse
        r2  = $r2
        """)

    end for
    new Plot (null, y, yp, "IGD for Perceptron y", lines = true)

end simplePerceptron2

