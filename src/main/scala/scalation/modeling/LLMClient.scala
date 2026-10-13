
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Sat Aug 01 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Provider-Agnostic LLM Completion Clients
 *
 *  Defines the minimal `LLMClient` trait that LLM-driven searches depend on,
 *  with six implementations:
 *      `OpenAIClient`     -- OpenAI chat-completions endpoint (cloud, needs a key)
 *      `AnthropicClient`  -- Anthropic messages endpoint (cloud, needs a key)
 *      `LocalClient`      -- ANY OpenAI-chat-completions-COMPATIBLE local server
 *                            (Ollama, vLLM, llama.cpp's llama-server, LM Studio):
 *                            a genuine LLM, but OFFLINE -- no network egress, no
 *                            API key, no per-token cost, no rate limit.  This is
 *                            what makes the "reduced LLM dependence" arms of the
 *                            paper (llm-diag, llm-gen) fully reproducible on a
 *                            single machine with no cloud account at all.
 *      `CachedClient`     -- disk cache wrapper keyed by SHA-256 of the prompt;
 *                            gives free re-runs, exact reproducibility (ship the
 *                            cache directory as a paper artifact), and a token
 *                            cost ledger.  NOTE: with temperature > 0 the FIRST
 *                            population of the cache is still stochastic -- the
 *                            cache freezes a run, it does not make the API
 *                            deterministic; report over multiple cached seeds.
 *      `CountingClient`   -- counts consultations, estimated tokens and cost so
 *                            experiments can report LLM cost per run; wrap it
 *                            OUTSIDE `CachedClient` to count calls the cache
 *                            answered too
 *      `MockLLMClient`    -- canned responses for offline UNIT TESTS (fixed
 *                            strings, not a real model -- distinct from
 *                            `LocalClient`, which is a real model running
 *                            offline; @see `sbt test`, which must never need a
 *                            network, an API key, OR a local server)
 *
 *  No third-party dependency: the request bodies are rendered and the replies
 *  read by the small `Json` object below (ScalaTion carries no external JSON
 *  library), so this file compiles with the rest of the project as is.
 *  API keys are read from environment variables (OPENAI_API_KEY,
 *  ANTHROPIC_API_KEY) -- NEVER hard-code keys in source or commit them.
 *
 *  @note    AI Assisted Code
 */

package scalation
package modeling

import java.io.IOException
import java.net.URI
import java.net.http.{HttpClient, HttpRequest, HttpResponse}
import java.nio.file.{Files, Path}
import java.security.MessageDigest
import java.time.Duration

import scala.collection.mutable.{ArrayBuffer => VEC}


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LLMClient` trait abstracts a large language model completion service,
 *  so the search loop is independent of the provider (OpenAI, Anthropic, local,
 *  cached, or mocked).
 */
trait LLMClient:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the model's completion for the given prompts.
     *  @param system  the system prompt (task contract, e.g., the formula grammar)
     *  @param user    the user prompt (dataset schema, scored population, request)
     */
    def complete (system: String, user: String): String

end LLMClient


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `OpenAIClient` class calls the OpenAI chat-completions endpoint.
 *  @param model  the model name (e.g., "gpt-4o", "gpt-4o-mini")
 *  @param temp   the sampling temperature (higher => more diverse proposals)
 */
class OpenAIClient (model: String = "gpt-4o-mini", temp: Double = 0.8)
      extends LLMClient:

    private val key = sys.env.getOrElse ("OPENAI_API_KEY",
                          throw new IllegalStateException ("environment variable OPENAI_API_KEY not set"))

    private val client = HttpClient.newBuilder ().connectTimeout (Duration.ofSeconds (30)).build ()

    def complete (system: String, user: String): String =
        val body = Json.chatBody (model, temp, system, user)

        val request = HttpRequest.newBuilder ()
            .uri (URI.create ("https://api.openai.com/v1/chat/completions"))
            .header ("Content-Type",  "application/json")
            .header ("Authorization", s"Bearer $key")
            .timeout (Duration.ofSeconds (120))
            .POST (HttpRequest.BodyPublishers.ofString (body)).build ()

        val response = LLMClient_.sendWithRetry (client, request)
        Json.string (response, "content")                                 // choices[0].message.content
    end complete

end OpenAIClient


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `AnthropicClient` class calls the Anthropic messages endpoint.
 *  @param model      the model name (e.g., "claude-sonnet-4-5")
 *  @param temp       the sampling temperature
 *  @param maxTokens  the response token cap (required by the API)
 */
class AnthropicClient (model: String = "claude-sonnet-4-5", temp: Double = 0.8,
                       maxTokens: Int = 2048)
      extends LLMClient:

    private val key = sys.env.getOrElse ("ANTHROPIC_API_KEY",
                          throw new IllegalStateException ("environment variable ANTHROPIC_API_KEY not set"))

    private val client = HttpClient.newBuilder ().connectTimeout (Duration.ofSeconds (30)).build ()

    def complete (system: String, user: String): String =
        val body = Json.obj ("model"       -> Json.str (model),
                             "temperature" -> Json.num (temp),
                             "max_tokens"  -> maxTokens.toString,
                             "system"      -> Json.str (system),          // top-level, not a message
                             "messages"    -> Json.arr (Json.obj ("role" -> Json.str ("user"),
                                                                  "content" -> Json.str (user))))

        val request = HttpRequest.newBuilder ()
            .uri (URI.create ("https://api.anthropic.com/v1/messages"))
            .header ("Content-Type",      "application/json")
            .header ("x-api-key",         key)
            .header ("anthropic-version", "2023-06-01")
            .timeout (Duration.ofSeconds (120))
            .POST (HttpRequest.BodyPublishers.ofString (body)).build ()

        val response = LLMClient_.sendWithRetry (client, request)
        Json.string (response, "text")                                    // content[0].text
    end complete

end AnthropicClient


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LocalClient` class calls ANY locally-hosted server that speaks the
 *  OpenAI chat-completions wire format -- Ollama, vLLM, llama.cpp's
 *  llama-server, and LM Studio all do.  This is a REAL model (unlike
 *  `MockLLMClient`), but it never leaves the machine: no network egress
 *  beyond localhost, no API key, no per-token billing, no rate limit.  For
 *  the paper this is what makes the reduced-LLM-dependence arms (llm-diag,
 *  llm-gen -- one seed call plus a few budgeted escapes) verifiable end-to-end
 *  with zero cloud dependency, and it removes provider nondeterminism as a
 *  confound when comparing update RULES rather than update SOURCES.
 *  Default endpoint/model target Ollama (`ollama pull llama3.1 && ollama serve`,
 *  which exposes an OpenAI-compatible route at /v1/chat/completions); pass a
 *  different endpoint/model for vLLM (commonly :8000) or llama-server
 *  (commonly :8080).  Most local servers accept requests with no
 *  Authorization header at all; `apiKeyEnv` is offered only for servers
 *  configured to require one (e.g., vLLM started with --api-key).
 *  @param endpoint   the chat-completions URL of the local server
 *  @param model      the model name as registered with that server
 *  @param temp       the sampling temperature (higher => more diverse proposals)
 *  @param apiKeyEnv  the ENV VAR holding an optional local API key ("" => omit
 *                    the Authorization header entirely, the common case)
 *  @param timeoutMs  the per-request timeout (local inference can be slow on
 *                    CPU-only hardware, hence a generous default)
 */
class LocalClient (endpoint: String = "http://localhost:11434/v1/chat/completions",
                   model: String = "llama3.1", temp: Double = 0.8,
                   apiKeyEnv: String = "", timeoutMs: Int = 180000)
      extends LLMClient:

    private val key    = if apiKeyEnv.isEmpty then None else sys.env.get (apiKeyEnv)
    private val client = HttpClient.newBuilder ().connectTimeout (Duration.ofSeconds (5)).build ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether a server is listening at `endpoint` (any HTTP response,
     *  including a 404/405 on a bare GET, counts -- only a connection failure
     *  or timeout means "not reachable").  Experiment drivers should call this
     *  BEFORE constructing an `LLMProposer` around a `LocalClient`, so a run
     *  without a local server running degrades to the zero-LLM arms instead
     *  of throwing mid-search.
     *  @param probeMs  the connect/read timeout for the reachability probe
     */
    def reachable (probeMs: Int = 500): Boolean =
        try
            val probe = HttpClient.newBuilder ().connectTimeout (Duration.ofMillis (probeMs)).build ()
            val req   = HttpRequest.newBuilder ().uri (URI.create (endpoint))
                            .timeout (Duration.ofMillis (probeMs)).GET ().build ()
            probe.send (req, HttpResponse.BodyHandlers.discarding ())
            true
        catch case _: Exception => false
    end reachable

    def complete (system: String, user: String): String =
        val body = Json.chatBody (model, temp, system, user)

        val builder = HttpRequest.newBuilder ()
            .uri (URI.create (endpoint))
            .header ("Content-Type", "application/json")
            .timeout (Duration.ofMillis (timeoutMs))
        key.foreach (k => builder.header ("Authorization", s"Bearer $k"))
        val request = builder.POST (HttpRequest.BodyPublishers.ofString (body)).build ()

        val response = LLMClient_.sendWithRetry (client, request)
        Json.string (response, "content")                                 // choices[0].message.content
    end complete

end LocalClient


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CachedClient` class wraps any `LLMClient` with a disk cache keyed by a
 *  SHA-256 hash of (tag, system, user): a hit reads from disk (no network); a
 *  miss calls through and persists.  Enables offline re-runs and reproducible
 *  experiments (@see file header note on stochastic first population).  Wraps
 *  `LocalClient` just as well as the cloud clients -- caching a local model's
 *  replies still buys instant re-runs and an exact shipped transcript.
 *  The `tag` MUST identify the model (and sampling temperature) behind
 *  `inner`: the prompts alone do not, so two clients that differ only in the
 *  model would otherwise share cache entries and a "stronger LLM" ablation
 *  would silently replay the weaker model's completions.  An empty tag hashes
 *  exactly as before, so existing cache directories stay valid.
 *  @param inner  the wrapped client (called only on cache misses)
 *  @param dir    the cache directory (created if absent)
 *  @param tag    the model/temperature identity folded into the cache key
 *                (e.g. "gpt-4o-mini@0.8"); "" => prompts-only key (legacy)
 */
class CachedClient (inner: LLMClient, dir: String = "llm_cache", tag: String = "")
      extends LLMClient:

    def complete (system: String, user: String): String =
        val digest = MessageDigest.getInstance ("SHA-256")                // length-prefix system so
                         .digest (s"$tag${system.length}:$system$user".getBytes ("UTF-8"))  // (a,bc) can't collide with (ab,c)
                         .map ("%02x".format (_)).mkString
        val file = Path.of (dir, s"$digest.txt")
        if Files.exists (file) then Files.readString (file)
        else
            val out = inner.complete (system, user)
            Files.createDirectories (file.getParent)
            Files.writeString (file, out)
            out
    end complete

end CachedClient


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CountingClient` class wraps any `LLMClient` and counts consultations,
 *  prompt and reply characters (tokens are estimated as characters / 4) and
 *  the resulting cost, so experiments can report LLM cost per run.  Wrap it
 *  OUTSIDE `CachedClient`, as in CountingClient (CachedClient (...)), to count
 *  the method's consultations whether or not the cache answered them.
 *  @param inner     the wrapped client
 *  @param priceIn   the price of prompt tokens in USD per million
 *  @param priceOut  the price of reply tokens in USD per million
 */
class CountingClient (inner: LLMClient, priceIn: Double = 0.0, priceOut: Double = 0.0)
      extends LLMClient:

    private var _calls    = 0                                         // completions requested
    private var _inChars  = 0L                                        // prompt characters
    private var _outChars = 0L                                        // reply characters

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of completions requested so far.
     */
    def calls: Int = _calls

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the total prompt + reply characters so far.
     */
    def chars: Long = _inChars + _outChars

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the estimated total tokens so far (characters / 4).
     */
    def tokens: Long = chars / 4

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the estimated cost so far in USD.
     */
    def cost: Double = (_inChars / 4.0) * priceIn / 1e6 + (_outChars / 4.0) * priceOut / 1e6

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Count and delegate one completion.
     *  @param system  the system prompt
     *  @param user    the user prompt
     */
    def complete (system: String, user: String): String =
        _calls   += 1
        _inChars += system.length + user.length
        val out = inner.complete (system, user)
        _outChars += out.length
        out
    end complete

end CountingClient


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MockLLMClient` class returns canned responses in order (cycling), for
 *  offline unit tests of the parser + search loop -- `sbt test` must never need
 *  a network, an API key, or a local server.
 *  @param responses  the canned responses to return, in order
 */
class MockLLMClient (responses: VEC [String])
      extends LLMClient:

    private var i = 0

    def complete (system: String, user: String): String =
        val out = responses (i % responses.size); i += 1; out
    end complete

end MockLLMClient


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Json` object renders the request bodies the chat endpoints expect and
 *  reads one string field out of their replies -- all this file needs, so no
 *  JSON library is required.
 */
object Json:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Render a string as a JSON string literal (quoted, escaped).
     *  @param s  the string
     */
    def str (s: String): String =
        val sb = new StringBuilder ("\"")
        for c <- s do
            c match
            case '"'  => sb.append ("\\\"")
            case '\\' => sb.append ("\\\\")
            case '\n' => sb.append ("\\n")
            case '\r' => sb.append ("\\r")
            case '\t' => sb.append ("\\t")
            case _ if c < ' ' => sb.append (f"\\u${c.toInt}%04x")
            case _    => sb.append (c)
        end for
        sb.append ('"').toString
    end str

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Render a number.
     *  @param d  the number
     */
    def num (d: Double): String = d.toString

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Render an object from (key, rendered value) pairs.
     *  @param fields  the fields (values already rendered)
     */
    def obj (fields: (String, String)*): String =
        fields.map ((k, v) => s"${str (k)}: $v").mkString ("{", ", ", "}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Render an array from rendered items.
     *  @param items  the items (already rendered)
     */
    def arr (items: String*): String = items.mkString ("[", ", ", "]")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Render a chat-completions body (OpenAI and compatible local servers).
     *  @param model   the model name
     *  @param temp    the sampling temperature
     *  @param system  the system prompt
     *  @param user    the user prompt
     */
    def chatBody (model: String, temp: Double, system: String, user: String): String =
        obj ("model"       -> str (model),
             "temperature" -> num (temp),
             "messages"    -> arr (obj ("role" -> str ("system"), "content" -> str (system)),
                                   obj ("role" -> str ("user"),   "content" -> str (user))))
    end chatBody

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the value of the FIRST string field with the given key in a JSON
     *  text (e.g., "content" for choices[0].message.content of a chat reply,
     *  "text" for content[0].text of a messages reply); throw when absent.
     *  A key inside a string value is not matched (it is not followed by ':').
     *  @param json  the JSON text
     *  @param key   the field name
     */
    def string (json: String, key: String): String =
        val pat  = "\"" + key + "\""
        var from = 0
        while from >= 0 do
            val i = json.indexOf (pat, from)
            if i < 0 then from = -1
            else
                var j = i + pat.length
                while j < json.length && json(j).isWhitespace do j += 1
                if j < json.length && json(j) == ':' then
                    j += 1
                    while j < json.length && json(j).isWhitespace do j += 1
                    if j < json.length && json(j) == '"' then return unquote (json, j + 1)
                from = i + pat.length
        end while
        throw new RuntimeException (s"Json.string: no string field '$key' in reply: ${json.take (200)}")
    end string

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Decode a JSON string literal starting after its opening quote.
     *  @param json   the JSON text
     *  @param start  the index of the first character after the opening quote
     */
    private def unquote (json: String, start: Int): String =
        val sb = new StringBuilder ()
        var j  = start
        while j < json.length && json(j) != '"' do
            val c = json(j)
            if c == '\\' && j + 1 < json.length then
                json(j+1) match
                    case '"'  => sb.append ('"');  j += 2
                    case '\\' => sb.append ('\\'); j += 2
                    case '/'  => sb.append ('/');  j += 2
                    case 'n'  => sb.append ('\n'); j += 2
                    case 'r'  => sb.append ('\r'); j += 2
                    case 't'  => sb.append ('\t'); j += 2
                    case 'b'  => sb.append ('\b'); j += 2
                    case 'f'  => sb.append ('\f'); j += 2
                    case 'u' if j + 5 < json.length =>
                        sb.append (Integer.parseInt (json.substring (j+2, j+6), 16).toChar); j += 6
                    case other => sb.append (other); j += 2
            else
                sb.append (c); j += 1
        end while
        sb.toString
    end unquote

end Json


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LLMClient_` object provides shared HTTP plumbing for the API clients.
 */
object LLMClient_ :

    private val debug = debugf ("LLMClient_", true)                   // debug function
    private val MAX_TRIES = 3                                         // attempts per request

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Send the request, retrying (up to `MAX_TRIES` attempts, with a growing
     *  pause) on a dropped connection, a rate-limit (429) or a server (5xx)
     *  error; throw on any other non-200 status or when the attempts run out.
     *  A long live grid makes hundreds of calls, and a single dropped HTTP/2
     *  connection must not abort the run.
     *  @param client   the shared HTTP client
     *  @param request  the fully-built request
     */
    def sendWithRetry (client: HttpClient, request: HttpRequest): String =
        var attempt = 0
        var body: String = null
        while body == null do
            attempt += 1
            try
                val response = client.send (request, HttpResponse.BodyHandlers.ofString ())
                val code     = response.statusCode ()
                if code == 200 then body = response.body ()
                else if (code == 429 || code >= 500) && attempt < MAX_TRIES then
                    debug ("sendWithRetry", s"status $code; retrying in ${5 * attempt}s")
                    Thread.sleep (5000L * attempt)
                else throw new RuntimeException (s"LLM HTTP $code: ${response.body ()}")
            catch case e: IOException =>
                if attempt >= MAX_TRIES then throw e
                debug ("sendWithRetry", s"${e.getMessage}; retrying in ${5 * attempt}s")
                Thread.sleep (5000L * attempt)
        end while
        body
    end sendWithRetry

end LLMClient_


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lLMClientTest` main function tests the mock, cache and counting clients
 *  OFFLINE (no network, no API key), verifying the caching and counting contracts.
 *  > runMain scalation.modeling.lLMClientTest
 */
@main def lLMClientTest (): Unit =

    banner ("MockLLMClient: canned responses in order")
    val mock = new MockLLMClient (VEC ("y ~= y~(3,1) + hosp~1", "y ~= y~(5,1) + wave(2)"))
    assert (mock.complete ("s", "u1") == "y ~= y~(3,1) + hosp~1")
    assert (mock.complete ("s", "u2") == "y ~= y~(5,1) + wave(2)")
    println ("mock: ordered responses OK")

    banner ("CachedClient: second identical call must NOT reach the inner client")
    var calls = 0
    val counting = new LLMClient:
        def complete (system: String, user: String): String = { calls += 1; s"response-$calls" }
    val dir    = Files.createTempDirectory ("llm_cache_test").toString
    val cached = new CachedClient (counting, dir)

    val a = cached.complete ("sys", "prompt-A")
    val b = cached.complete ("sys", "prompt-A")                       // cache hit
    val c = cached.complete ("sys", "prompt-B")                       // different key: miss
    assert (a == b,     "cache must return the stored response")
    assert (calls == 2, s"inner must be called twice (A miss, B miss), was $calls")
    assert (c == "response-2")
    println ("cache: hit/miss contract OK")

    banner ("CachedClient: a different model tag must NOT share entries")
    val cachedB = new CachedClient (counting, dir, tag = "other-model@0.8")
    val d = cachedB.complete ("sys", "prompt-A")                      // same prompts, other tag: miss
    assert (calls == 3, s"inner must be called for the new tag (A miss under tag), was $calls")
    assert (d == "response-3" && d != a, "tagged entry must be distinct from the untagged one")
    println ("cache: tag isolation OK")

    banner ("CountingClient: counts every consultation, cache hits included")
    val counter = new CountingClient (cached, priceIn = 1.0, priceOut = 4.0)
    counter.complete ("sys", "prompt-A")                              // cache hit: counted, inner not called
    counter.complete ("sys", "prompt-C")                              // miss: counted, inner called
    assert (counter.calls == 2, s"counter must count both calls, was ${counter.calls}")
    assert (calls == 4,         s"inner must be called once more (C miss), was $calls")
    assert (counter.chars > 0 && counter.tokens == counter.chars / 4, "tokens must be characters / 4")
    assert (counter.cost > 0.0, "cost must follow from tokens and prices")
    println (f"counting: calls, tokens (${counter.tokens}) and cost ($$${counter.cost}%.8f) OK")

    banner ("Json: request bodies render and reply fields read back (no JSON library)")
    val sysP = "grammar:\n  y ~= y~1 + \"quoted\" \\ backslash\ttab"
    val body = Json.chatBody ("gpt-4o-mini", 0.8, sysP, "user text")
    println (body)
    assert (Json.string (body, "content") == sysP, "the first content field is the system prompt, escapes round-trip")
    val reply = "{\"id\":\"x\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\"," +
                "\"content\":\"y ~= y~(3,1) + x1~2\\nTFORM: Log1p\\n\\u00e9\"},\"finish_reason\":\"stop\"}]," +
                "\"usage\":{\"prompt_tokens\":10}}"
    assert (Json.string (reply, "content") == "y ~= y~(3,1) + x1~2\nTFORM: Log1p\n\u00e9", "chat reply content decoded")
    val reply2 = "{\"type\":\"message\",\"content\":[{\"type\":\"text\",\"text\":\"hello\"}]}"
    assert (Json.string (reply2, "text") == "hello", "messages reply text decoded (\"type\":\"text\" is a value, not the key)")
    println ("json: render/read contract OK")

    banner ("live/local clients (skipped unless keys are set or a server is running)")
    println (s"OPENAI_API_KEY set:    ${sys.env.contains ("OPENAI_API_KEY")}")
    println (s"ANTHROPIC_API_KEY set: ${sys.env.contains ("ANTHROPIC_API_KEY")}")
    val local = new LocalClient ()
    println (s"local server reachable at ${"http://localhost:11434/v1/chat/completions"}: ${local.reachable ()}")

end lLMClientTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lLMClientTest2` main function makes ONE live OpenAI call to verify the
 *  API key end-to-end (network, auth, billing).  Requires OPENAI_API_KEY.
 *  > runMain scalation.modeling.lLMClientTest2
 */
@main def lLMClientTest2 (): Unit =

    banner ("live OpenAI smoke test (one small call)")
    val llm   = new OpenAIClient ("gpt-4o-mini", temp = 0.0)
    val reply = llm.complete ("You reply with exactly one word.", "Say READY.")
    println (s"reply = '$reply'")

end lLMClientTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lLMClientTest3` main function makes ONE OFFLINE call to a local model
 *  server, to verify the endpoint/model end-to-end -- no network egress
 *  beyond localhost, no API key, no billing.  Skips gracefully (prints a
 *  how-to-start message) if no server is reachable at the default endpoint.
 *  Start a server first, e.g. one of:
 *      ollama pull llama3.1 && ollama serve                (port 11434)
 *      python -m vllm.entrypoints.openai.api_server \
 *             --model <hf-model-id> --port 8000             (port 8000)
 *      llama-server -m <model.gguf> --port 8080              (port 8080)
 *  > runMain scalation.modeling.lLMClientTest3
 */
@main def lLMClientTest3 (): Unit =

    banner ("offline local-model smoke test (one small call)")
    val llm = new LocalClient ()                                      // defaults target Ollama on :11434
    if ! llm.reachable () then
        println ("no local server reachable at http://localhost:11434/v1/chat/completions")
        println ("start one first, e.g.:  ollama pull llama3.1 && ollama serve")
        println ("(for vLLM or llama-server, pass a matching endpoint/model to LocalClient)")
    else
        val reply = llm.complete ("You reply with exactly one word.", "Say READY.")
        println (s"reply = '$reply'")

end lLMClientTest3
