// tool_call.cpp — lenient parser + deterministic dispatcher for model-chosen
// tasks (home-ops#584, M2).
//
// The parser is intentionally forgiving about *packaging* (fences, prose,
// wrapping quotes) and strict about *content* (known tool, typed and in-range
// arguments). That split is what makes the validity number meaningful: it
// measures whether the model picked a usable call, not whether it avoided
// markdown.
#include "tool_call.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace crispasr {
namespace voice {
namespace {

// ---------------------------------------------------------------------------
// tiny helpers
// ---------------------------------------------------------------------------
bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v';
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && is_space(s[a])) ++a;
    while (b > a && is_space(s[b - 1])) --b;
    return s.substr(a, b - a);
}

std::string lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    return s;
}

void replace_all(std::string* s, const std::string& what, const std::string& with) {
    size_t pos = 0;
    while ((pos = s->find(what, pos)) != std::string::npos) {
        s->replace(pos, what.size(), with);
        pos += with.size();
    }
}

// Fold separators so "device-status", "Device Status" and "device_status"
// are the same name.
std::string fold_name(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    bool last_sep = false;
    for (char c : trim(in)) {
        if (c == '-' || c == '_' || is_space(c)) {
            if (!out.empty() && !last_sep) out.push_back('_');
            last_sep = true;
        } else {
            out.push_back((char)std::tolower((unsigned char)c));
            last_sep = false;
        }
    }
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out;
}

// ---------------------------------------------------------------------------
// minimal JSON value (objects, arrays, strings, numbers, bool, null)
// ---------------------------------------------------------------------------
struct JsonValue {
    enum Kind { Null, Bool, Number, String, Array, Object };
    Kind kind = Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    const JsonValue* get(const std::string& key) const {
        for (const auto& kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

class JsonParser {
   public:
    JsonParser(const std::string& s) : s_(s) {}

    bool parse_value(JsonValue* out, int depth) {
        if (depth > 8) return false;
        skip_ws();
        if (i_ >= s_.size()) return false;
        const char c = s_[i_];
        if (c == '{') return parse_object(out, depth);
        if (c == '[') return parse_array(out, depth);
        if (c == '"') {
            out->kind = JsonValue::String;
            return parse_string(&out->str);
        }
        if (c == 't' || c == 'f') {
            if (s_.compare(i_, 4, "true") == 0) {
                i_ += 4;
                out->kind = JsonValue::Bool;
                out->b = true;
                return true;
            }
            if (s_.compare(i_, 5, "false") == 0) {
                i_ += 5;
                out->kind = JsonValue::Bool;
                out->b = false;
                return true;
            }
            return false;
        }
        if (s_.compare(i_, 4, "null") == 0) {
            i_ += 4;
            out->kind = JsonValue::Null;
            return true;
        }
        return parse_number(out);
    }

    bool at_end_after_ws() {
        skip_ws();
        return i_ >= s_.size();
    }

   private:
    void skip_ws() {
        while (i_ < s_.size() && is_space(s_[i_])) ++i_;
    }

    bool parse_object(JsonValue* out, int depth) {
        out->kind = JsonValue::Object;
        ++i_;  // '{'
        skip_ws();
        if (i_ < s_.size() && s_[i_] == '}') {
            ++i_;
            return true;
        }
        while (true) {
            skip_ws();
            std::string key;
            if (i_ >= s_.size() || s_[i_] != '"') return false;
            if (!parse_string(&key)) return false;
            skip_ws();
            if (i_ >= s_.size() || s_[i_] != ':') return false;
            ++i_;
            JsonValue v;
            if (!parse_value(&v, depth + 1)) return false;
            out->obj.emplace_back(std::move(key), std::move(v));
            skip_ws();
            if (i_ >= s_.size()) return false;
            if (s_[i_] == ',') {
                ++i_;
                continue;
            }
            if (s_[i_] == '}') {
                ++i_;
                return true;
            }
            return false;
        }
    }

    bool parse_array(JsonValue* out, int depth) {
        out->kind = JsonValue::Array;
        ++i_;  // '['
        skip_ws();
        if (i_ < s_.size() && s_[i_] == ']') {
            ++i_;
            return true;
        }
        while (true) {
            JsonValue v;
            if (!parse_value(&v, depth + 1)) return false;
            out->arr.push_back(std::move(v));
            skip_ws();
            if (i_ >= s_.size()) return false;
            if (s_[i_] == ',') {
                ++i_;
                continue;
            }
            if (s_[i_] == ']') {
                ++i_;
                return true;
            }
            return false;
        }
    }

    bool parse_string(std::string* out) {
        if (i_ >= s_.size() || s_[i_] != '"') return false;
        ++i_;
        out->clear();
        while (i_ < s_.size()) {
            const char c = s_[i_++];
            if (c == '"') return true;
            if (c != '\\') {
                out->push_back(c);
                continue;
            }
            if (i_ >= s_.size()) return false;
            const char e = s_[i_++];
            switch (e) {
                case '"': out->push_back('"'); break;
                case '\\': out->push_back('\\'); break;
                case '/': out->push_back('/'); break;
                case 'b': out->push_back('\b'); break;
                case 'f': out->push_back('\f'); break;
                case 'n': out->push_back('\n'); break;
                case 'r': out->push_back('\r'); break;
                case 't': out->push_back('\t'); break;
                case 'u': {
                    // Enough for names and the odd accented character; a
                    // surrogate pair collapses to '?', which no tool argument
                    // depends on.
                    if (i_ + 4 > s_.size()) return false;
                    unsigned cp = 0;
                    for (int k = 0; k < 4; ++k) {
                        const char h = s_[i_ + (size_t)k];
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
                        else return false;
                    }
                    i_ += 4;
                    if (cp < 0x80) {
                        out->push_back((char)cp);
                    } else if (cp < 0x800) {
                        out->push_back((char)(0xC0 | (cp >> 6)));
                        out->push_back((char)(0x80 | (cp & 0x3F)));
                    } else {
                        out->push_back((char)(0xE0 | (cp >> 12)));
                        out->push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                        out->push_back((char)(0x80 | (cp & 0x3F)));
                    }
                    break;
                }
                default: return false;
            }
        }
        return false;  // unterminated
    }

    bool parse_number(JsonValue* out) {
        const size_t start = i_;
        if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
        bool digits = false;
        while (i_ < s_.size() && std::isdigit((unsigned char)s_[i_])) {
            ++i_;
            digits = true;
        }
        if (i_ < s_.size() && s_[i_] == '.') {
            ++i_;
            while (i_ < s_.size() && std::isdigit((unsigned char)s_[i_])) {
                ++i_;
                digits = true;
            }
        }
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            ++i_;
            if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
            while (i_ < s_.size() && std::isdigit((unsigned char)s_[i_])) ++i_;
        }
        if (!digits) return false;
        const std::string tok = s_.substr(start, i_ - start);
        char* end = nullptr;
        out->num = std::strtod(tok.c_str(), &end);
        if (!end || *end != '\0' || !std::isfinite(out->num)) return false;
        out->kind = JsonValue::Number;
        return true;
    }

    const std::string& s_;
    size_t i_ = 0;
};

// First balanced {...} in `s`, honouring string literals and escapes. Empty
// when there is no balanced object (prose, a bare name, or truncated JSON).
std::string first_balanced_object(const std::string& s) {
    const size_t start = s.find('{');
    if (start == std::string::npos) return std::string();
    int depth = 0;
    bool in_str = false;
    bool esc = false;
    for (size_t i = start; i < s.size(); ++i) {
        const char c = s[i];
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') {
            in_str = true;
            continue;
        }
        if (c == '{') {
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0) return s.substr(start, i - start + 1);
            if (depth < 0) return std::string();
        }
    }
    return std::string();  // unterminated
}

// Strips one layer of wrapping quotes and undoes the escaping inside, so a
// model that replies with a JSON *string* still works.
std::string strip_wrapping_quotes(const std::string& in) {
    if (in.size() < 2 || in.front() != '"' || in.back() != '"') return in;
    std::string out;
    out.reserve(in.size() - 2);
    for (size_t i = 1; i + 1 < in.size(); ++i) {
        if (in[i] == '\\' && i + 2 < in.size()) {
            const char e = in[i + 1];
            if (e == '"' || e == '\\' || e == '/') {
                out.push_back(e);
                ++i;
                continue;
            }
        }
        out.push_back(in[i]);
    }
    return out;
}

// Integer coercion for `minutes`: a JSON number that is integral, or a numeric
// string ("5", " 5 ", "5.0"). Anything else is a type error, not a default.
bool coerce_int(const JsonValue& v, int* out) {
    if (v.kind == JsonValue::Number) {
        if (std::fabs(v.num) > 1.0e9) return false;
        const double r = std::floor(v.num + 0.5);
        if (std::fabs(v.num - r) > 1.0e-9) return false;
        *out = (int)r;
        return true;
    }
    if (v.kind == JsonValue::String) {
        const std::string t = trim(v.str);
        if (t.empty()) return false;
        char* end = nullptr;
        const double d = std::strtod(t.c_str(), &end);
        if (!end || end == t.c_str()) return false;
        while (*end && is_space(*end)) ++end;
        if (*end != '\0') return false;
        if (!std::isfinite(d) || std::fabs(d) > 1.0e9) return false;
        const double r = std::floor(d + 0.5);
        if (std::fabs(d - r) > 1.0e-9) return false;
        *out = (int)r;
        return true;
    }
    return false;
}

// Lenient device value: a string, folded, with the obvious synonyms accepted.
bool coerce_device(const JsonValue& v, std::string* out) {
    if (v.kind != JsonValue::String) return false;
    std::string d = lower(trim(v.str));
    replace_all(&d, "the ", "");
    replace_all(&d, " ", "_");
    if (d == "garage" || d == "garage_door" || d == "garagedoor" ||
        d == "door")
        *out = "garage";
    else if (d == "lights" || d == "light" || d == "lamp" || d == "lamps")
        *out = "lights";
    else if (d == "thermostat" || d == "heat" || d == "heating" ||
             d == "temperature")
        *out = "thermostat";
    else
        return false;
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// names
// ---------------------------------------------------------------------------
const char* to_string(ToolId id) {
    switch (id) {
        case ToolId::DeviceStatus: return "device_status";
        case ToolId::SetTimer: return "set_timer";
        case ToolId::Smalltalk: return "smalltalk";
        default: return "unknown";
    }
}

ToolId tool_from_name(const std::string& name) {
    const std::string n = fold_name(name);
    if (n == "device_status" || n == "devicestatus" || n == "device" ||
        n == "status")
        return ToolId::DeviceStatus;
    if (n == "set_timer" || n == "settimer" || n == "timer")
        return ToolId::SetTimer;
    if (n == "smalltalk" || n == "small_talk" || n == "chat")
        return ToolId::Smalltalk;
    return ToolId::Unknown;
}

const char* to_string(ToolMode m) {
    switch (m) {
        case ToolMode::Off: return "off";
        case ToolMode::Llm: return "llm";
        case ToolMode::FakeValid: return "fake_valid";
        case ToolMode::FakeInvalid: return "fake_invalid";
    }
    return "llm";
}

bool parse_tool_mode(const std::string& name, ToolMode* out) {
    const std::string n = fold_name(name);
    if (n.empty() || n == "llm" || n == "model") {
        *out = ToolMode::Llm;
        return true;
    }
    if (n == "off" || n == "none" || n == "disabled") {
        *out = ToolMode::Off;
        return true;
    }
    if (n == "fake_valid" || n == "valid") {
        *out = ToolMode::FakeValid;
        return true;
    }
    if (n == "fake_invalid" || n == "invalid") {
        *out = ToolMode::FakeInvalid;
        return true;
    }
    return false;
}

ToolMode tool_mode_from_env() {
    const char* v = std::getenv("VOICE_TOOL_MODE");
    ToolMode m = ToolMode::Llm;
    if (v && *v && !parse_tool_mode(v, &m)) return ToolMode::Llm;
    return m;
}

// ---------------------------------------------------------------------------
// parser
// ---------------------------------------------------------------------------
ToolCall parse_tool_call(const std::string& text) {
    ToolCall call;
    call.raw = trim(text);
    if (call.raw.empty()) {
        call.reason = "empty reply";
        return call;
    }

    // Packaging: drop code fences and stray backticks, unwrap quotes. Anything
    // that survives is content, which is what gets validated.
    std::string s = call.raw;
    replace_all(&s, "```", "");
    s = trim(s);
    s = trim(strip_wrapping_quotes(s));

    const std::string obj_text = first_balanced_object(s);
    if (obj_text.empty()) {
        // A bare tool name is the most common non-JSON reply a small model
        // produces. A text containing '{' is a broken object, not a name.
        if (s.find('{') != std::string::npos) {
            call.reason = "unterminated JSON object";
            return call;
        }
        std::string bare = trim(strip_wrapping_quotes(s));
        while (!bare.empty() && (bare.back() == '.' || bare.back() == '!' ||
                                 bare.back() == ',' || bare.back() == ':'))
            bare.pop_back();
        const ToolId id = tool_from_name(bare);
        if (id == ToolId::Unknown) {
            call.reason = "no JSON object and no bare tool name";
            return call;
        }
        call.tool = id;
        call.valid = true;
        call.reason = "bare tool name";
        if (id == ToolId::DeviceStatus) call.device = "garage";
        if (id == ToolId::SetTimer) {
            call.minutes = 5;
            call.reason = "bare tool name, minutes defaulted to 5";
        }
        return call;
    }

    JsonValue root;
    JsonParser jp(obj_text);
    if (!jp.parse_value(&root, 0) || root.kind != JsonValue::Object) {
        call.reason = "JSON is not an object (or is malformed)";
        return call;
    }

    const JsonValue* tool = root.get("tool");
    if (tool == nullptr || tool->kind != JsonValue::String) {
        // Some models use "name"/"action". Cheap and unambiguous.
        tool = root.get("name");
        if (tool == nullptr || tool->kind != JsonValue::String)
            tool = root.get("action");
    }
    if (tool == nullptr || tool->kind != JsonValue::String) {
        call.reason = "no string \"tool\" field";
        return call;
    }
    call.tool = tool_from_name(tool->str);
    if (call.tool == ToolId::Unknown) {
        call.reason = "unknown tool \"" + tool->str + "\"";
        return call;
    }

    const JsonValue* args = root.get("args");
    if (args == nullptr) args = root.get("arguments");
    if (args != nullptr && args->kind != JsonValue::Object) {
        call.reason = "\"args\" is not an object";
        return call;
    }

    switch (call.tool) {
        case ToolId::DeviceStatus: {
            const JsonValue* dev = args ? args->get("device") : nullptr;
            if (dev == nullptr) {
                call.reason = "device_status without \"device\"";
                return call;
            }
            if (!coerce_device(*dev, &call.device)) {
                call.reason = "\"device\" is not garage|lights|thermostat";
                return call;
            }
            call.valid = true;
            return call;
        }
        case ToolId::SetTimer: {
            const JsonValue* min = args ? args->get("minutes") : nullptr;
            if (min == nullptr) {
                call.reason = "set_timer without \"minutes\"";
                return call;
            }
            if (!coerce_int(*min, &call.minutes)) {
                call.reason = "\"minutes\" is not an integer";
                return call;
            }
            if (call.minutes < 1 || call.minutes > 600) {
                call.reason = "\"minutes\" outside 1..600";
                return call;
            }
            call.valid = true;
            return call;
        }
        case ToolId::Smalltalk:
            // No arguments by definition; extra keys are ignored rather than
            // treated as a schema error (a chatty model may add a "reply").
            call.valid = true;
            return call;
        default:
            call.reason = "unknown tool";
            return call;
    }
}

// ---------------------------------------------------------------------------
// prompt
// ---------------------------------------------------------------------------
std::vector<ChatMessage> tool_call_messages(const std::string& transcript) {
    ChatMessage sys;
    sys.role = "system";
    sys.content =
        "You are a tool router. Choose exactly one tool for the request and "
        "reply with one JSON object and nothing else.\n"
        "Tools:\n"
        "{\"tool\":\"device_status\",\"args\":{\"device\":\"garage\"}}\n"
        "  device is one of garage, lights, thermostat. Slow check, about 6 "
        "seconds.\n"
        "{\"tool\":\"set_timer\",\"args\":{\"minutes\":5}}\n"
        "  minutes is an integer from 1 to 600. Answers immediately.\n"
        "{\"tool\":\"smalltalk\",\"args\":{}}\n"
        "  greetings, thanks, opinions, chit-chat: no device check and no "
        "timer.\n"
        "Rules: one object, double quotes, no prose, no markdown, no code "
        "fences, no explanation. Never invent a tool name. If the request is "
        "unclear, use device_status for the garage.";

    ChatMessage user;
    user.role = "user";
    user.content = "Request: \"" + transcript + "\"\nReply with the one JSON object.";
    return {sys, user};
}

std::string fake_tool_reply(ToolMode mode, const std::string& transcript) {
    if (mode == ToolMode::FakeInvalid) {
        // The kind of reply a small model gives when it ignores the format:
        // helpful prose, no call. The parser must reject it and the loop must
        // fall back, not crash.
        return "I think I should check the garage door for you before answering.";
    }
    const std::string t = lower(transcript);
    auto has = [&](const char* needle) {
        return t.find(needle) != std::string::npos;
    };
    if (has("timer") || has("minute") || has("remind")) {
        int minutes = 5;
        for (size_t i = 0; i < t.size(); ++i) {
            if (std::isdigit((unsigned char)t[i])) {
                minutes = std::atoi(t.c_str() + i);
                break;
            }
        }
        if (minutes < 1 || minutes > 600) minutes = 5;
        return "{\"tool\":\"set_timer\",\"args\":{\"minutes\":" +
               std::to_string(minutes) + "}}";
    }
    if (has("hello") || has(" hey") || has("thank") || has("how are you") ||
        has("good morning") || has("good evening") || has("joke")) {
        return "{\"tool\":\"smalltalk\",\"args\":{}}";
    }
    if (has("thermostat") || has("temperature") || has("heat"))
        return "{\"tool\":\"device_status\",\"args\":{\"device\":\"thermostat\"}}";
    if (has("light") || has("lamp"))
        return "{\"tool\":\"device_status\",\"args\":{\"device\":\"lights\"}}";
    return "{\"tool\":\"device_status\",\"args\":{\"device\":\"garage\"}}";
}

std::string tool_task_request(const ToolCall& call, const std::string& transcript) {
    switch (call.tool) {
        case ToolId::DeviceStatus:
            // Unchanged on purpose: the simulated result and the answer prompt
            // stay comparable with the pre-M2 runs.
            return transcript;
        case ToolId::SetTimer:
            return "timer set for " + std::to_string(call.minutes) +
                   " minutes (\"" + transcript + "\")";
        case ToolId::Smalltalk:
            return "no task needed (\"" + transcript + "\")";
        default:
            return transcript;
    }
}

// ---------------------------------------------------------------------------
// profiles + dispatcher
// ---------------------------------------------------------------------------
ToolProfile tool_profile(ToolId tool, const Config& cfg) {
    ToolProfile p;
    switch (tool) {
        case ToolId::DeviceStatus:
            p.latency_ms = cfg.task_ms;
            p.jitter_ms = cfg.task_jitter_ms;
            p.fail_percent = cfg.task_fail_percent;
            break;
        case ToolId::SetTimer:
            // "Answers immediately": well under the ack, so the answer is the
            // only speech the user hears.
            p.latency_ms = 800;
            p.jitter_ms = 200;
            p.fail_percent = 0;
            break;
        case ToolId::Smalltalk:
            p.latency_ms = 0;
            p.jitter_ms = 0;
            p.fail_percent = 0;
            break;
        default:
            p.latency_ms = cfg.task_ms;
            p.jitter_ms = cfg.task_jitter_ms;
            p.fail_percent = cfg.task_fail_percent;
            break;
    }
    if (p.latency_ms < 0) p.latency_ms = 0;
    return p;
}

struct ToolDispatcher::Impl {
    // Indexed by ToolId for the three real tools; Unknown maps to slot 0 but is
    // never dispatched (the loop falls back before calling).
    SimulatedTaskAdapter* adapters[3] = {nullptr, nullptr, nullptr};

    static int slot(ToolId t) {
        const int s = (int)t;
        return (s >= 0 && s <= 2) ? s : 0;
    }
};

ToolDispatcher::ToolDispatcher(const Config& cfg) : impl_(new Impl()) {
    for (int i = 0; i < 3; ++i) {
        const ToolProfile p = tool_profile((ToolId)i, cfg);
        impl_->adapters[i] =
            new SimulatedTaskAdapter(p.latency_ms, p.jitter_ms, p.fail_percent);
    }
}

ToolDispatcher::~ToolDispatcher() {
    shutdown();
    delete impl_;
    impl_ = nullptr;
}

int ToolDispatcher::dispatch(const ToolCall& call, const std::string& transcript,
                             TaskCallback cb) {
    if (impl_ == nullptr) return 0;
    const int slot = Impl::slot(call.tool);
    if (impl_->adapters[slot] == nullptr) return 0;
    const int local = impl_->adapters[slot]->dispatch(
        tool_task_request(call, transcript), std::move(cb));
    if (local <= 0) return 0;
    // Global id: local ids restart at 1 per adapter, and tool ids are encoded
    // in the low three bits so cancel() needs no map in the orchestrator.
    return local * 8 + slot;
}

void ToolDispatcher::cancel(int id) {
    if (impl_ == nullptr || id <= 0) return;
    const int slot = id % 8;
    if (slot < 0 || slot > 2) return;
    if (impl_->adapters[slot] == nullptr) return;
    impl_->adapters[slot]->cancel(id / 8);
}

void ToolDispatcher::shutdown() {
    if (impl_ == nullptr) return;
    for (int i = 0; i < 3; ++i) {
        if (impl_->adapters[i] != nullptr) {
            impl_->adapters[i]->shutdown();
            delete impl_->adapters[i];
            impl_->adapters[i] = nullptr;
        }
    }
}

}  // namespace voice
}  // namespace crispasr
