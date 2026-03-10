
// build.sbt

lazy val scalation = project.in(file("."))
  .settings(
    scalaVersion  := "3.8.2",
    scalacOptions ++= Seq(
       "-deprecation",         // emit warning and location for usages of deprecated APIs
       "-explain",             // explain errors in more detail
//     "-explain-types",       // explain type errors in more detail
       "-new-syntax",          // require `then` and `do` in control expressions.
       "-Wunused:all",         // warn of unused imports, ...
       "-Werror")     // fail the compilation if there are any warnings
//  javacOptions  += "--add-modules jdk.incubator.vector"
  )

fork := true

// resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"
// resolvers += Opts.resolver.sonatypeSnapshots
// resolvers += Opts.resolver.sonatypeOssSnapshots

// ScalaFx (2D and 3D Graphics)
// https://mvnrepository.com/artifact/org.scalafx/scalafx_3/22.0.0-R33
// libraryDependencies += "org.scalafx" %% "scalafx" % "21.0.0-R32"
libraryDependencies += "org.scalafx" %% "scalafx" % "22.0.0-R33"

// Gson (json)
// https://mvnrepository.com/artifact/com.google.code.gson/gson
// libraryDependencies += "com.google.code.gson" % "gson" % "2.10"

// JUnit (unit testing)
// https://mvnrepository.com/artifact/org.junit.jupiter/junit-jupiter-api
// libraryDependencies += "org.junit.jupiter" % "junit-jupiter-api" % "5.9.1" % Test

