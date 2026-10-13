
// build.sbt

lazy val scalation = project.in(file("."))
  .settings(
    scalaVersion  := "3.9.0",
    scalacOptions ++= Seq(
       "-deprecation",              // emit warning and location for usages of deprecated APIs
       "-explain",                  // explain errors in more detail
//     "-explain-types",            // explain type errors in more detail
       "-new-syntax",               // require `then` and `do` in control expressions
       "-encoding", "UTF-8",        // source files are UTF-8 (compile-time)
       "-Wunused:all",              // warn of unused imports, ...
       "-Werror"),                  // fail the compilation if there are any warnings
    fork := true,
    outputStrategy := Some(StdoutOutput),
    javaOptions ++= Seq(
       "-Dfile.encoding=UTF-8",     // default charset (runtime)
       "-Dstdout.encoding=UTF-8",   // JDK 18+ : console stream encoding
       "-Dstderr.encoding=UTF-8"),
    libraryDependencies += "org.scala-lang.modules" %% "scala-parallel-collections" % "1.2.0",

    // ujson (json) -- used by scalation.modeling.LLMClient (LLMSR4TS)
    // https://mvnrepository.com/artifact/com.lihaoyi/ujson
    libraryDependencies += "com.lihaoyi" %% "ujson" % "4.0.2",
    libraryDependencies += "com.lihaoyi" %% "fastparse" % "3.1.1"
  )





// resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"
// resolvers += Opts.resolver.sonatypeSnapshots
// resolvers += Opts.resolver.sonatypeOssSnapshots

// ScalaFx (2D and 3D Graphics)
// https://mvnrepository.com/artifact/org.scalafx/scalafx_3/22.0.0-R33
// libraryDependencies += "org.scalafx" %% "scalafx" % "21.0.0-R32"
// libraryDependencies += "org.scalafx" %% "scalafx" % "22.0.0-R33"

// Gson (json)
// https://mvnrepository.com/artifact/com.google.code.gson/gson
// libraryDependencies += "com.google.code.gson" % "gson" % "2.10"

// JUnit (unit testing)
// https://mvnrepository.com/artifact/org.junit.jupiter/junit-jupiter-api
// libraryDependencies += "org.junit.jupiter" % "junit-jupiter-api" % "5.9.1" % Test
