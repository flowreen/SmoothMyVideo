// Barebones Python 3.14 bindings for TensorRT-RTX 1.5.0.114.
//
// NVIDIA ships tensorrt_rtx wheels only for cp38..cp313 and does not publish the binding
// source; this module reimplements, with pybind11, exactly the API subset that
// SmoothMyVideo's engine/trt_runtime.py uses (same names and semantics as the official
// wheel), so `import tensorrt_rtx as trt` works unchanged on CPython 3.14:
//
//   trt.__version__
//   trt.Logger(trt.Logger.WARNING)
//   trt.DataType.{FLOAT,HALF,INT32,INT64,BOOL,BF16}
//   trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED       (int()-able)
//   trt.MemoryPoolType.WORKSPACE
//   trt.TensorIOMode.{INPUT,OUTPUT}
//   trt.Builder(logger).create_network(flags) / .create_builder_config()
//                      .build_serialized_network(net, cfg) -> bytes | None
//   trt.OnnxParser(net, logger).parse_from_file(p) / .num_errors / .get_error(i)
//   config.set_memory_pool_limit(pool, size)
//   trt.Runtime(logger).deserialize_cuda_engine(bytes) -> engine
//   engine.num_io_tensors / .get_tensor_name(i) / .get_tensor_dtype(n) /
//         .get_tensor_shape(n) / .get_tensor_mode(n) /
//         .create_runtime_config() / .create_execution_context([runtime_config])
//   runtime_config.create_runtime_cache() / .set_runtime_cache(cache)
//   cache.serialize() -> bytes / .deserialize(bytes) / .reset()
//   context.set_input_shape(n, shape) / .set_tensor_address(n, ptr) /
//          .execute_async_v3(stream_handle)
//
// Object lifetimes follow TensorRT 10 rules (plain `delete`); py::keep_alive ties
// dependents to their factories (context->engine->runtime, network->builder, ...).
#include <pybind11/pybind11.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstdio>
#include <string>

namespace py = pybind11;
using namespace nvinfer1;

namespace {

class PyLogger : public ILogger {
public:
    explicit PyLogger(Severity minSev) : mMin(minSev) {}
    void log(Severity sev, char const* msg) noexcept override {
        if (static_cast<int>(sev) > static_cast<int>(mMin)) return;
        char const* tag = sev == Severity::kINTERNAL_ERROR ? "INTERNAL ERROR"
                        : sev == Severity::kERROR          ? "ERROR"
                        : sev == Severity::kWARNING        ? "WARNING"
                        : sev == Severity::kINFO           ? "INFO" : "VERBOSE";
        std::fprintf(stderr, "[TRT-RTX %s] %s\n", tag, msg);
        std::fflush(stderr);
    }
    Severity mMin;
};

Dims seqToDims(py::sequence const& s) {
    Dims d{};
    d.nbDims = static_cast<int32_t>(py::len(s));
    if (d.nbDims > Dims::MAX_DIMS) throw std::runtime_error("too many dimensions");
    for (int32_t i = 0; i < d.nbDims; ++i) d.d[i] = s[i].cast<int64_t>();
    return d;
}

py::tuple dimsToTuple(Dims const& d) {
    py::tuple t(d.nbDims);
    for (int32_t i = 0; i < d.nbDims; ++i) t[i] = py::int_(d.d[i]);
    return t;
}

// IHostMemory -> bytes (copies, then frees the TRT-owned blob)
py::object hostMemToBytes(IHostMemory* m) {
    if (!m) return py::none();
    py::bytes b(static_cast<char const*>(m->data()), m->size());
    delete m;
    return b;
}

}  // namespace

PYBIND11_MODULE(tensorrt_rtx, mod) {
    mod.doc() = "Barebones TensorRT-RTX bindings for CPython 3.14 (SmoothMyVideo subset)";
    mod.attr("__version__") = std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR)
        + "." + std::to_string(NV_TENSORRT_PATCH) + "." + std::to_string(NV_TENSORRT_BUILD);

    py::enum_<DataType>(mod, "DataType")
        .value("FLOAT", DataType::kFLOAT)
        .value("HALF", DataType::kHALF)
        .value("INT32", DataType::kINT32)
        .value("BOOL", DataType::kBOOL)
        .value("BF16", DataType::kBF16)
        .value("INT64", DataType::kINT64);

    py::enum_<NetworkDefinitionCreationFlag>(mod, "NetworkDefinitionCreationFlag")
        .value("STRONGLY_TYPED", NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);

    py::enum_<MemoryPoolType>(mod, "MemoryPoolType")
        .value("WORKSPACE", MemoryPoolType::kWORKSPACE);

    py::enum_<TensorIOMode>(mod, "TensorIOMode")
        .value("NONE", TensorIOMode::kNONE)
        .value("INPUT", TensorIOMode::kINPUT)
        .value("OUTPUT", TensorIOMode::kOUTPUT);

    auto logger = py::class_<PyLogger>(mod, "Logger");
    py::enum_<ILogger::Severity>(logger, "Severity")  // register BEFORE use as a default argument
        .value("INTERNAL_ERROR", ILogger::Severity::kINTERNAL_ERROR)
        .value("ERROR", ILogger::Severity::kERROR)
        .value("WARNING", ILogger::Severity::kWARNING)
        .value("INFO", ILogger::Severity::kINFO)
        .value("VERBOSE", ILogger::Severity::kVERBOSE);
    logger.def(py::init<ILogger::Severity>(), py::arg("min_severity") = ILogger::Severity::kWARNING);
    logger.attr("INTERNAL_ERROR") = ILogger::Severity::kINTERNAL_ERROR;
    logger.attr("ERROR") = ILogger::Severity::kERROR;
    logger.attr("WARNING") = ILogger::Severity::kWARNING;
    logger.attr("INFO") = ILogger::Severity::kINFO;
    logger.attr("VERBOSE") = ILogger::Severity::kVERBOSE;

    py::class_<INetworkDefinition>(mod, "INetworkDefinition");
    py::class_<IBuilderConfig>(mod, "IBuilderConfig")
        .def("set_memory_pool_limit",
             [](IBuilderConfig& c, MemoryPoolType pool, size_t size) { c.setMemoryPoolLimit(pool, size); },
             py::arg("pool"), py::arg("pool_size"));

    py::class_<IBuilder>(mod, "Builder")
        .def(py::init([](PyLogger& l) {
                 IBuilder* b = createInferBuilder(l);
                 if (!b) throw std::runtime_error("createInferBuilder failed");
                 return b;
             }),
             py::arg("logger"), py::keep_alive<1, 2>())
        .def("create_network",
             [](IBuilder& b, uint32_t flags) { return b.createNetworkV2(flags); },
             py::arg("flags") = 0U, py::return_value_policy::take_ownership, py::keep_alive<0, 1>())
        .def("create_builder_config",
             [](IBuilder& b) { return b.createBuilderConfig(); },
             py::return_value_policy::take_ownership, py::keep_alive<0, 1>())
        .def("build_serialized_network",
             [](IBuilder& b, INetworkDefinition& n, IBuilderConfig& c) {
                 IHostMemory* m;
                 {
                     py::gil_scoped_release nogil;
                     m = b.buildSerializedNetwork(n, c);
                 }
                 return hostMemToBytes(m);
             },
             py::arg("network"), py::arg("config"));

    py::class_<nvonnxparser::IParser>(mod, "OnnxParser")
        .def(py::init([](INetworkDefinition& n, PyLogger& l) {
                 auto* p = nvonnxparser::createParser(n, l);
                 if (!p) throw std::runtime_error("createParser failed");
                 return p;
             }),
             py::arg("network"), py::arg("logger"), py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("parse_from_file",
             [](nvonnxparser::IParser& p, std::string const& path) {
                 py::gil_scoped_release nogil;
                 return p.parseFromFile(path.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
             },
             py::arg("model"))
        .def_property_readonly("num_errors", [](nvonnxparser::IParser& p) { return p.getNbErrors(); })
        .def("get_error", [](nvonnxparser::IParser& p, int i) {
            auto const* e = p.getError(i);
            return std::string(e ? e->desc() : "unknown error");
        });

    py::class_<IRuntimeCache>(mod, "IRuntimeCache")
        .def("serialize", [](IRuntimeCache const& c) { return hostMemToBytes(c.serialize()); })
        .def("deserialize",
             [](IRuntimeCache& c, py::bytes blob) {
                 std::string_view sv = blob;
                 return c.deserialize(sv.data(), sv.size());
             },
             py::arg("blob"))
        .def("reset", &IRuntimeCache::reset);

    py::class_<IRuntimeConfig>(mod, "IRuntimeConfig")
        .def("create_runtime_cache",
             [](IRuntimeConfig& rc) { return rc.createRuntimeCache(); },
             py::return_value_policy::take_ownership, py::keep_alive<0, 1>())
        .def("set_runtime_cache",
             [](IRuntimeConfig& rc, IRuntimeCache& c) {
                 if (!rc.setRuntimeCache(c)) throw std::runtime_error("setRuntimeCache failed");
             },
             py::arg("cache"), py::keep_alive<1, 2>());

    py::class_<IExecutionContext>(mod, "IExecutionContext")
        .def("set_input_shape",
             [](IExecutionContext& c, std::string const& name, py::sequence shape) {
                 if (!c.setInputShape(name.c_str(), seqToDims(shape)))
                     throw std::runtime_error("set_input_shape failed for " + name);
             },
             py::arg("name"), py::arg("shape"))
        .def("set_tensor_address",
             [](IExecutionContext& c, std::string const& name, uintptr_t ptr) {
                 if (!c.setTensorAddress(name.c_str(), reinterpret_cast<void*>(ptr)))
                     throw std::runtime_error("set_tensor_address failed for " + name);
             },
             py::arg("name"), py::arg("memory"))
        .def("execute_async_v3",
             [](IExecutionContext& c, uintptr_t stream) {
                 py::gil_scoped_release nogil;
                 return c.enqueueV3(reinterpret_cast<cudaStream_t>(stream));
             },
             py::arg("stream_handle"));

    py::class_<ICudaEngine>(mod, "ICudaEngine")
        .def_property_readonly("num_io_tensors", &ICudaEngine::getNbIOTensors)
        .def("get_tensor_name",
             [](ICudaEngine& e, int32_t i) { return std::string(e.getIOTensorName(i)); },
             py::arg("index"))
        .def("get_tensor_dtype",
             [](ICudaEngine& e, std::string const& n) { return e.getTensorDataType(n.c_str()); },
             py::arg("name"))
        .def("get_tensor_shape",
             [](ICudaEngine& e, std::string const& n) { return dimsToTuple(e.getTensorShape(n.c_str())); },
             py::arg("name"))
        .def("get_tensor_mode",
             [](ICudaEngine& e, std::string const& n) { return e.getTensorIOMode(n.c_str()); },
             py::arg("name"))
        .def("create_runtime_config",
             [](ICudaEngine& e) {
                 IRuntimeConfig* rc = e.createRuntimeConfig();
                 if (!rc) throw std::runtime_error("createRuntimeConfig failed");
                 return rc;
             },
             py::return_value_policy::take_ownership, py::keep_alive<0, 1>())
        .def("create_execution_context",
             [](ICudaEngine& e) {
                 IExecutionContext* c;
                 {
                     py::gil_scoped_release nogil;
                     c = e.createExecutionContext();
                 }
                 if (!c) throw std::runtime_error("createExecutionContext failed");
                 return c;
             },
             py::return_value_policy::take_ownership, py::keep_alive<0, 1>())
        .def("create_execution_context",
             [](ICudaEngine& e, IRuntimeConfig& rc) {
                 IExecutionContext* c;
                 {
                     py::gil_scoped_release nogil;
                     c = e.createExecutionContext(&rc);
                 }
                 if (!c) throw std::runtime_error("createExecutionContext(runtime_config) failed");
                 return c;
             },
             py::arg("runtime_config"), py::return_value_policy::take_ownership,
             py::keep_alive<0, 1>(), py::keep_alive<0, 2>());

    py::class_<IRuntime>(mod, "Runtime")
        .def(py::init([](PyLogger& l) {
                 IRuntime* r = createInferRuntime(l);
                 if (!r) throw std::runtime_error("createInferRuntime failed");
                 return r;
             }),
             py::arg("logger"), py::keep_alive<1, 2>())
        .def("deserialize_cuda_engine",
             [](IRuntime& r, py::bytes blob) {
                 std::string_view sv = blob;
                 ICudaEngine* e;
                 {
                     py::gil_scoped_release nogil;
                     e = r.deserializeCudaEngine(sv.data(), sv.size());
                 }
                 if (!e) throw std::runtime_error("deserialize_cuda_engine failed");
                 return e;
             },
             py::arg("serialized_engine"), py::return_value_policy::take_ownership,
             py::keep_alive<0, 1>());
}
