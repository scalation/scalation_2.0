<h1 align="center">ScalaTion</h1>

<p align="center"><b>SCALAble SimulaTION</b> &mdash; a Scala-based system for Simulation, Optimization and Analytics</p>

<p align="center">
  <a href="https://scalation.github.io/scalation_2.0/"><img alt="Scaladoc" src="https://img.shields.io/badge/docs-Scaladoc-blue"></a>
  <a href="https://www.scala-lang.org/download/"><img alt="Scala 3.8.3" src="https://img.shields.io/badge/Scala-3.8.3-DC322F?logo=scala&logoColor=white"></a>
  <a href="https://jdk.java.net/archive/"><img alt="Java 21" src="https://img.shields.io/badge/Java-21-007396"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> &middot;
  <a href="#source-packages">Packages</a> &middot;
  <a href="https://scalation.github.io/scalation_2.0/">Scaladoc</a> &middot;
  <a href="#papers">Papers</a> &middot;
  <a href="#license">License</a>
</p>

---

**Welcome to ScalaTion, the Scala-based system for Simulation, Optimization and Analytics.**

This system, coded in Scala, supports multi-paradigm simulation modeling including
*tableau*, *event*, *process*, *dynamics*, *dynamics_pde*, *activity* and *state* oriented models.

[Scala](https://www.scala-lang.org) is a modern object-oriented, functional programming language
that is well-suited for developing simulation engines.
It is in the Java family of languages and can call Java code.
The inclusion of advanced and functional programming capabilities
makes the code much more concise than Java.
ScalaTion also supports optimization and analytics, as they nowadays go hand-in-hand with simulation modeling.
Some of the algorithms come in both sequential and parallel versions.

**ScalaTion 2.0** defines multiple `.par` subpackages that contain parallel versions of sequential algorithms.
Future directions include completing the `scala3d` and `physics` packages.

## Requirements

| Tool | Version | Notes |
|---|---|---|
| [Java](https://www.oracle.com/java/technologies/downloads/) | 21 | or an OpenJDK 21 from the [OpenJDK archive](https://jdk.java.net/archive/) |
| [sbt](https://www.scala-sbt.org/) | 1.x | recommended build tool; the version used is pinned in `project/build.properties` and fetched by sbt |
| [Scala](https://www.scala-lang.org/download/) | 3.8.3 | set in `build.sbt`; sbt downloads the compiler, so a separate install is optional |

New to sbt? See [sbt: Hello, World](https://www.scala-sbt.org/1.x/docs/Hello.html).

## Quick start

**1. Clone the repository**

```bash
git clone https://github.com/scalation/scalation_2.0.git
```

**2. Build all the modules** &mdash; change into the ScalaTion base directory and run sbt

```bash
cd scalation_2.0
sbt compile
```

**3. Compile code or run apps** &mdash; enter sbt and type `compile`, `runMain` or `exit`

```console
$ sbt
sbt:scalation> compile
sbt:scalation> runMain scalation.modeling.simpleRegressionTest
sbt:scalation> runMain scalation.modeling.regressionTest
sbt:scalation> exit
```

The `compile` command is optional, since `runMain` compiles first if need be.
A single app can also be run straight from the shell:

```bash
sbt "runMain scalation.modeling.regressionTest"
```

The demo and test apps are `@main` methods named after their class with a lower-case first letter and the
suffix `Test` (`Regression` &rarr; `regressionTest`, `regressionTest2`, &hellip;).

> [!NOTE]
> `build.sbt` compiles with `-Werror`, so any compiler warning (for example, an unused import) fails the build.

## Repository layout

ScalaTion is set up to use the Simple Build Tool [sbt](https://www.scala-sbt.org), with the build
specification [`build.sbt`](build.sbt).

| Path | Contents |
|---|---|
| [`src/main/scala/scalation`](src/main/scala/scalation) | the source packages |
| [`docs`](docs) | the generated Scaladoc, published at <https://scalation.github.io/scalation_2.0/> |
| [`data`](data) | data files for input, output and analytics (in subdirectories) |
| [`store`](store) | database files |
| `lib` | unmanaged jar files, if any (sbt puts every jar in `lib` on the classpath); see [`where_is_jar.txt`](where_is_jar.txt) |
| [`build.sbt`](build.sbt), [`project`](project) | the sbt build |

To regenerate the Scaladoc in `docs`:

```bash
sbt clean compile doc
rm -rf docs && cp -r target/scala-3.8.3/api docs
```

## Using ScalaTion from your own project

To develop projects that use ScalaTion but live outside ScalaTion:

**1. Build the ScalaTion jar**

```bash
cd scalation_2.0
sbt package          # writes target/scala-3.8.3/scalation_3-0.1.0-SNAPSHOT.jar
```

**2. Create your project and copy the jar into its `lib` directory**

```bash
cd your_project_dir
sbt new scala/scala3.g8          # enter a project name when prompted, e.g. hello
cd hello
mkdir lib
cp <dir>/scalation_2.0/target/scala-3.8.3/scalation_3-0.1.0-SNAPSHOT.jar lib/
```

where `<dir>` is the directory you placed `scalation_2.0` in.

**3. Match ScalaTion's Scala version and dependency in your `build.sbt`**

```scala
scalaVersion := "3.8.3"
libraryDependencies += "org.scala-lang.modules" %% "scala-parallel-collections" % "1.2.0"
```

(An unmanaged jar does not bring its own dependencies.) Then write your code under `src/main/scala`
and start it with `sbt run`.

## IDE support

- [IntelliJ IDEA](https://www.jetbrains.com/idea/download) with the
  [Scala plugin](https://www.jetbrains.com/help/idea/discover-intellij-idea-for-scala.html):
  open the `scalation_2.0` folder and import it as an sbt project.
- [VS Code](https://code.visualstudio.com/) with [Metals](https://scalameta.org/metals/):
  open the folder and import the build when prompted.

## Source packages

| Package | Description | Docs |
|---|---|---|
| [**scalation**](src/main/scala/scalation) | **The ScalaTion foundational packages.** | [Scaladoc](https://scalation.github.io/scalation_2.0/scalation.html) |
| [animation](src/main/scala/scalation/animation) | The `animation` package supports the animation of models. | [Scaladoc](https://scalation.github.io/scalation_2.0/scalation/animation.html) |
| [calculus](src/main/scala/scalation/calculus) | The `calculus` package supports numerical differentiation and integration. | [Scaladoc](https://scalation.github.io/scalation_2.0/scalation/calculus.html) |
| [database](src/main/scala/scalation/database) | The `database` package supports relational and graph databases. | [Scaladoc](https://scalation.github.io/scalation_2.0/scalation/database.html) |
| [dynamics](src/main/scala/scalation/dynamics) | The `dynamics` package supports the development of ODE models. | [Scaladoc](https://scalation.github.io/scalation_2.0/scalation/dynamics.html) |
| [mathstat](src/main/scala/scalation/mathstat) | The `mathstat` package supports basic math and statistics. | [Scaladoc](https://scalation.github.io/scalation_2.0/scalation/mathstat.html) |
| [modeling](src/main/scala/scalation/modeling) | The `modeling` package supports the development of several types of data science models. | [Scaladoc](https://scalation.github.io/scalation_2.0/scalation/modeling.html) |
| [optimization](src/main/scala/scalation/optimization) | The `optimization` package supports linear and nonlinear optimization. | [Scaladoc](https://scalation.github.io/scalation_2.0/scalation/optimization.html) |
| [random](src/main/scala/scalation/random) | The `random` package supports random variate generation. | [Scaladoc](https://scalation.github.io/scalation_2.0/scalation/random.html) |
| [scala2d](src/main/scala/scalation/scala2d) | The `scala2d` package supports simple 2D graphics in Scala, based upon `java.swing`, `java.awt` and `java.awt.geom`. | [Scaladoc](https://scalation.github.io/scalation_2.0/scalation/scala2d.html) |
| [scala3d](src/main/scala/scalation/scala3d) | The `scala3d` package (under development) explores JavaFX/ScalaFX as a lightweight 3D solution. | &mdash; |
| [simulation](src/main/scala/scalation/simulation) | The `simulation` package supports the development of simulation models. | [Scaladoc](https://scalation.github.io/scalation_2.0/scalation/simulation.html) |
| [theory](src/main/scala/scalation/theory) | The `theory` package (under development) brings together the elements of modeling and theory. | [Scaladoc](https://scalation.github.io/scalation_2.0/scalation/theory.html) |

## Papers

1. John A. Miller, Jun Han and Maria Hybinette,
   [**"Using Domain Specific Languages for Modeling and Simulation: ScalaTion as a Case Study,"**](https://informs-sim.org/wsc10papers/067.pdf)
   *Proceedings of the 2010 ACM/IEEE Winter Simulation Conference (WSC'10)*,
   Baltimore, Maryland (December 2010) pp. 741-752.

2. Michael E. Cotterell, John A. Miller and Tom Horton,
   [**"Unicode in Domain-Specific Programming Languages for Modeling & Simulation: ScalaTion as a Case Study,"**](https://arxiv.org/abs/1112.1751)
   arXiv preprint arXiv:1112.1751 (December 2011) pp. 1-10.

3. Michael E. Cotterell, John A. Miller, Jun Han and Tom Horton,
   [**"Extending ScalaTion, a Domain-Specific Language for Modeling & Simulation, for Simulation Optimization,"**](https://cobweb.cs.uga.edu/~jam/scalation_papers/alasim/alasim_extended_abstract.pdf)
   *Proceedings of the AlaSim International Modeling and Simulation Conference & Exhibition (AlaSim'12)*,
   Huntsville, Alabama (May 2012) pp. 1-1.

4. Yung Long Li,
   [**"Evaluation of Parallel Implementations of Dense and Sparse Matrices for the ScalaTion Library,"**](https://cobweb.cs.uga.edu/~jam/home/theses/li_thesis/thesis/TR_Yung_Long_Li.pdf)
   Technical Report, University of Georgia (December 2012) pp. 1-60.

5. John A. Miller, Michael E. Cotterell and Stephen J. Buckley,
   [**"Supporting a Modeling Continuum in ScalaTion: From Predictive Analytics to Simulation Modeling,"**](https://informs-sim.org/wsc13papers/includes/files/104.pdf)
   *Proceedings of the 2013 ACM/IEEE Winter Simulation Conference (WSC'13)*,
   Washington, DC (December 2013) pp. 1191-1202.

6. Matthew Saltz, Ayushi Jain, Abhishek Kothari, Arash Fard, John A. Miller and Lakshmish Ramaswamy,
   [**"DualIso: An Algorithm for Subgraph Pattern Matching on Very Large Labeled Graphs,"**](http://www.thecloudcomputing.org/2014/AdvanceProgram-ICWS-SCC-CLOUD-MS-BigDataCongress-SERVICES-2014.pdf)
   *Proceedings of the 3rd IEEE International Congress on Big Data*
   ([BigData'14](http://www.ieeebigdata.org/2014)),
   Anchorage, Alaska (June-July 2014) pp. 498-505.
   Online [supplement](https://cobweb.cs.uga.edu/~jam/home/theses/jain_thesis/bigdata_2014/BigDataCong2014_DualIso_Supplement.pdf).

## License

ScalaTion is released under the MIT License &mdash; see [LICENSE](LICENSE).

Copyright &copy; 2009-2021, John A. Miller and the University of Georgia.
