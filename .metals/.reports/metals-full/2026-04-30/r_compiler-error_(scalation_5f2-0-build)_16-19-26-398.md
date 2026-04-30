error id: D61FB54159C790A19FF1DECA6EAC18AB
file:///C:/Users/youse/OneDrive/Documents/GitHub/scalation_2.0/build.sbt
### java.lang.NullPointerException: Cannot read the array length because "a" is null

occurred in the presentation compiler.



action parameters:
uri: file:///C:/Users/youse/OneDrive/Documents/GitHub/scalation_2.0/build.sbt
text:
```scala
import _root_.scala.xml.{TopScope=>$scope}
import _root_.sbt._
import _root_.sbt.Keys._
import _root_.sbt.nio.Keys._
import _root_.sbt.ScriptedPlugin.autoImport._, _root_.sbt.plugins.JUnitXmlReportPlugin.autoImport._, _root_.sbt.plugins.MiniDependencyTreePlugin.autoImport._, _root_.bloop.integrations.sbt.BloopPlugin.autoImport._
import _root_.sbt.plugins.IvyPlugin, _root_.sbt.plugins.JvmPlugin, _root_.sbt.plugins.CorePlugin, _root_.sbt.ScriptedPlugin, _root_.sbt.plugins.SbtPlugin, _root_.sbt.plugins.SemanticdbPlugin, _root_.sbt.plugins.JUnitXmlReportPlugin, _root_.sbt.plugins.Giter8TemplatePlugin, _root_.sbt.plugins.MiniDependencyTreePlugin, _root_.bloop.integrations.sbt.BloopPlugin

// build.sbt

lazy val scalation = project.in(file("."))
  .settings(
    scalaVersion  := "3.8.3",
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

outputStrategy := Some(StdoutOutput)

libraryDependencies += "org.scala-lang.modules" %% "scala-parallel-collections" % "1.2.0"

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


```


presentation compiler configuration:
Scala version: 2.12.21
Classpath:
<WORKSPACE>\project\.bloop\scalation_2-0-build\bloop-bsp-clients-classes\classes-Metals-pZ-85e8SSCyHNlS0LkMxsw== [missing ], <HOME>\AppData\Local\bloop\cache\semanticdb\com.sourcegraph.semanticdb-javac.0.11.2\semanticdb-javac-0.11.2.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\sbt\1.12.9\sbt-1.12.9.jar [exists ], <HOME>\.sbt\boot\scala-2.12.21\lib\scala-library.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\ch\epfl\scala\sbt-bloop_2.12_1.0\2.0.19\sbt-bloop_2.12_1.0-2.0.19.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\main_2.12\1.12.9\main_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\io_2.12\1.10.5\io_2.12-1.10.5.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\ch\epfl\scala\bloop-config_2.12\2.3.3\bloop-config_2.12-2.3.3.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\logic_2.12\1.12.9\logic_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\actions_2.12\1.12.9\actions_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\main-settings_2.12\1.12.9\main-settings_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\run_2.12\1.12.9\run_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\command_2.12\1.12.9\command_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\collections_2.12\1.12.9\collections_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\scripted-plugin_2.12\1.12.9\scripted-plugin_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\zinc-lm-integration_2.12\1.12.9\zinc-lm-integration_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\util-logging_2.12\1.12.9\util-logging_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-lang\modules\scala-xml_2.12\2.3.0\scala-xml_2.12-2.3.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\launcher-interface\1.5.2\launcher-interface-1.5.2.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\github\ben-manes\caffeine\caffeine\2.8.5\caffeine-2.8.5.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\io\get-coursier\lm-coursier-shaded_2.12\2.1.13\lm-coursier-shaded_2.12-2.1.13.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\apache\logging\log4j\log4j-api\2.25.3\log4j-api-2.25.3.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\apache\logging\log4j\log4j-core\2.25.3\log4j-core-2.25.3.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\apache\logging\log4j\log4j-slf4j-impl\2.25.3\log4j-slf4j-impl-2.25.3.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\librarymanagement-core_2.12\1.12.0\librarymanagement-core_2.12-1.12.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\librarymanagement-ivy_2.12\1.12.0\librarymanagement-ivy_2.12-1.12.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\compiler-interface\1.12.0\compiler-interface-1.12.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\zinc-compile_2.12\1.12.0\zinc-compile_2.12-1.12.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\swoval\file-tree-views\2.1.12\file-tree-views-2.1.12.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\github\plokhotnyuk\jsoniter-scala\jsoniter-scala-core_2.12\2.13.5.2\jsoniter-scala-core_2.12-2.13.5.2.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\lihaoyi\unroll-annotation_2.12\0.1.12\unroll-annotation_2.12-0.1.12.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\util-relation_2.12\1.12.9\util-relation_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\completion_2.12\1.12.9\completion_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\task-system_2.12\1.12.9\task-system_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\tasks_2.12\1.12.9\tasks_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\testing_2.12\1.12.9\testing_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\util-tracking_2.12\1.12.9\util-tracking_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\eed3si9n\sjson-new-core_2.12\0.10.1\sjson-new-core_2.12-0.10.1.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\eed3si9n\sjson-new-scalajson_2.12\0.10.1\sjson-new-scalajson_2.12-0.10.1.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\eed3si9n\gigahorse-apache-http_2.12\0.9.3\gigahorse-apache-http_2.12-0.9.3.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\jline\jline-terminal\3.27.1\jline-terminal-3.27.1.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\zinc-classpath_2.12\1.12.0\zinc-classpath_2.12-1.12.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\zinc-apiinfo_2.12\1.12.0\zinc-apiinfo_2.12-1.12.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\zinc_2.12\1.12.0\zinc_2.12-1.12.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\core-macros_2.12\1.12.9\core-macros_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\util-cache_2.12\1.12.9\util-cache_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\util-control_2.12\1.12.9\util-control_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\protocol_2.12\1.12.9\protocol_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\template-resolver\0.1\template-resolver-0.1.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\util-position_2.12\1.12.9\util-position_2.12-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\zinc-compile-core_2.12\1.12.0\zinc-compile-core_2.12-1.12.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\util-interface\1.12.9\util-interface-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\jline\jline\2.14.7-sbt-9a88bc413e2b34a4580c001c654d1a7f4f65bf18\jline-2.14.7-sbt-9a88bc413e2b34a4580c001c654d1a7f4f65bf18.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\jline\jline-terminal-jni\3.27.1\jline-terminal-jni-3.27.1.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\jline\jline-native\3.27.1\jline-native-3.27.1.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\lmax\disruptor\3.4.2\disruptor-3.4.2.jar [exists ], <HOME>\.sbt\boot\scala-2.12.21\lib\scala-reflect.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\checkerframework\checker-qual\3.4.1\checker-qual-3.4.1.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\google\errorprone\error_prone_annotations\2.4.0\error_prone_annotations-2.4.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-lang\modules\scala-collection-compat_2.12\2.14.0\scala-collection-compat_2.12-2.14.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\slf4j\slf4j-api\1.7.36\slf4j-api-1.7.36.jar [exists ], <HOME>\.sbt\boot\scala-2.12.21\lib\scala-compiler.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\github\mwiede\jsch\2.27.5\jsch-2.27.5.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\ivy\ivy\2.3.0-sbt-77cc781d727b367d3761f097d89f5a4762771d41\ivy-2.3.0-sbt-77cc781d727b367d3761f097d89f5a4762771d41.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\jline\jline-reader\3.27.1\jline-reader-3.27.1.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\jline\jline-builtins\3.27.1\jline-builtins-3.27.1.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\test-agent\1.12.9\test-agent-1.12.9.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\test-interface\1.0\test-interface-1.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\eed3si9n\shaded-scalajson_2.12\1.0.0-M4\shaded-scalajson_2.12-1.0.0-M4.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\eed3si9n\shaded-jawn-parser_2.12\1.3.2\shaded-jawn-parser_2.12-1.3.2.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\eed3si9n\gigahorse-core_2.12\0.9.3\gigahorse-core_2.12-0.9.3.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\eed3si9n\shaded-apache-httpclient5\0.9.3\shaded-apache-httpclient5-0.9.3.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\compiler-bridge_2.12\1.12.0\compiler-bridge_2.12-1.12.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\zinc-classfile_2.12\1.12.0\zinc-classfile_2.12-1.12.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\zinc-core_2.12\1.12.0\zinc-core_2.12-1.12.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\zinc-persist_2.12\1.12.0\zinc-persist_2.12-1.12.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\eed3si9n\sjson-new-murmurhash_2.12\0.10.1\sjson-new-murmurhash_2.12-0.10.1.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\ipcsocket\ipcsocket\1.6.3\ipcsocket-1.6.3.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-lang\modules\scala-parser-combinators_2.12\1.1.2\scala-parser-combinators_2.12-1.1.2.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\net\openhft\zero-allocation-hashing\0.16\zero-allocation-hashing-0.16.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\fusesource\jansi\jansi\2.4.1\jansi-2.4.1.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\jline\jline-style\3.27.1\jline-style-3.27.1.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\typesafe\ssl-config-core_2.12\0.6.1\ssl-config-core_2.12-0.6.1.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\reactivestreams\reactive-streams\1.0.3\reactive-streams-1.0.3.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\zinc-persist-core-assembly\1.12.0\zinc-persist-core-assembly-1.12.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\org\scala-sbt\sbinary_2.12\0.5.1\sbinary_2.12-0.5.1.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\net\java\dev\jna\jna\5.12.0\jna-5.12.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\net\java\dev\jna\jna-platform\5.12.0\jna-platform-5.12.0.jar [exists ], <HOME>\AppData\Local\Coursier\cache\v1\https\repo1.maven.org\maven2\com\typesafe\config\1.4.2\config-1.4.2.jar [exists ]
Options:
-deprecation -Wconf:cat=unused-nowarn:s -Wconf:cat=unused-nowarn:s -Xsource:3 -Yrangepos -Xplugin-require:semanticdb




#### Error stacktrace:

```
java.base/java.util.Arrays.sort(Arrays.java:1234)
	scala.tools.nsc.classpath.JFileDirectoryLookup.listChildren(DirectoryClassPath.scala:119)
	scala.tools.nsc.classpath.JFileDirectoryLookup.listChildren$(DirectoryClassPath.scala:103)
	scala.tools.nsc.classpath.DirectoryClassPath.listChildren(DirectoryClassPath.scala:314)
	scala.tools.nsc.classpath.DirectoryClassPath.listChildren(DirectoryClassPath.scala:314)
	scala.tools.nsc.classpath.DirectoryLookup.list(DirectoryClassPath.scala:84)
	scala.tools.nsc.classpath.DirectoryLookup.list$(DirectoryClassPath.scala:79)
	scala.tools.nsc.classpath.DirectoryClassPath.list(DirectoryClassPath.scala:314)
	scala.tools.nsc.classpath.AggregateClassPath.$anonfun$list$3(AggregateClassPath.scala:105)
	scala.collection.Iterator.foreach(Iterator.scala:943)
	scala.collection.Iterator.foreach$(Iterator.scala:943)
	scala.collection.AbstractIterator.foreach(Iterator.scala:1431)
	scala.collection.IterableLike.foreach(IterableLike.scala:74)
	scala.collection.IterableLike.foreach$(IterableLike.scala:73)
	scala.collection.AbstractIterable.foreach(Iterable.scala:56)
	scala.tools.nsc.classpath.AggregateClassPath.list(AggregateClassPath.scala:101)
	scala.tools.nsc.util.ClassPath.list(ClassPath.scala:36)
	scala.tools.nsc.util.ClassPath.list$(ClassPath.scala:36)
	scala.tools.nsc.classpath.AggregateClassPath.list(AggregateClassPath.scala:30)
	scala.tools.nsc.symtab.SymbolLoaders$PackageLoader.doComplete(SymbolLoaders.scala:298)
	scala.tools.nsc.symtab.SymbolLoaders$SymbolLoader.complete(SymbolLoaders.scala:250)
	scala.reflect.internal.Symbols$Symbol.completeInfo(Symbols.scala:1542)
	scala.reflect.internal.Symbols$Symbol.info(Symbols.scala:1514)
	scala.reflect.internal.Types$TypeRef.decls(Types.scala:2283)
	scala.tools.nsc.typechecker.Namers$Namer.enterPackage(Namers.scala:766)
	scala.tools.nsc.typechecker.Namers$Namer.dispatch$1(Namers.scala:289)
	scala.tools.nsc.typechecker.Namers$Namer.standardEnterSym(Namers.scala:302)
	scala.tools.nsc.typechecker.AnalyzerPlugins.pluginsEnterSym(AnalyzerPlugins.scala:479)
	scala.tools.nsc.typechecker.AnalyzerPlugins.pluginsEnterSym$(AnalyzerPlugins.scala:478)
	scala.meta.internal.pc.MetalsGlobal$MetalsInteractiveAnalyzer.pluginsEnterSym(MetalsGlobal.scala:85)
	scala.tools.nsc.typechecker.Namers$Namer.enterSym(Namers.scala:280)
	scala.tools.nsc.typechecker.Analyzer$namerFactory$$anon$1.apply(Analyzer.scala:48)
	scala.tools.nsc.Global$GlobalPhase.applyPhase(Global.scala:465)
	scala.tools.nsc.Global$Run.$anonfun$compileLate$3(Global.scala:1657)
	scala.tools.nsc.Global$Run.$anonfun$compileLate$2(Global.scala:1657)
	scala.tools.nsc.Global$Run.$anonfun$compileLate$2$adapted(Global.scala:1656)
	scala.collection.Iterator.foreach(Iterator.scala:943)
	scala.collection.Iterator.foreach$(Iterator.scala:943)
	scala.collection.AbstractIterator.foreach(Iterator.scala:1431)
	scala.tools.nsc.Global$Run.compileLate(Global.scala:1656)
	scala.tools.nsc.interactive.Global.parseAndEnter(Global.scala:654)
	scala.tools.nsc.interactive.Global.typeCheck(Global.scala:664)
	scala.meta.internal.pc.SemanticdbTextDocumentProvider.textDocument(SemanticdbTextDocumentProvider.scala:35)
	scala.meta.internal.pc.ScalaPresentationCompiler.$anonfun$semanticdbTextDocument$1(ScalaPresentationCompiler.scala:564)
	scala.meta.internal.pc.CompilerAccess.withSharedCompiler(CompilerAccess.scala:148)
	scala.meta.internal.pc.CompilerAccess.$anonfun$withInterruptableCompiler$1(CompilerAccess.scala:92)
	scala.meta.internal.pc.CompilerAccess.$anonfun$onCompilerJobQueue$1(CompilerAccess.scala:209)
	scala.meta.internal.pc.CompilerJobQueue$Job.run(CompilerJobQueue.scala:152)
	java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1144)
	java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:642)
	java.base/java.lang.Thread.run(Thread.java:1583)
```
#### Short summary: 

java.lang.NullPointerException: Cannot read the array length because "a" is null