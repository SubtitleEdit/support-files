#include "ggml.h"
#include "gguf.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

int main(int argc, char ** argv) {
    try {
        if (argc != 2 && argc != 3) {
            throw std::runtime_error("Usage: propainter-checkpoint MODEL.gguf [TENSOR-DUMP]");
        }

        ggml_context * storage = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(
            gguf_init_from_file(argv[1], {argc == 2, argc == 2 ? nullptr : &storage}), &gguf_free);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(storage, &ggml_free);
        if (!file) {
            throw std::runtime_error("Cannot load the GGUF checkpoint");
        }

        const auto architecture = gguf_find_key(file.get(), "general.architecture");
        if (architecture < 0 || gguf_get_kv_type(file.get(), architecture) != GGUF_TYPE_STRING) {
            throw std::runtime_error("Missing checkpoint architecture");
        }
        const std::string_view model = gguf_get_val_str(file.get(), architecture);
        if (model != "propainter-raft" && model != "propainter-flow-completion" && model != "propainter-generator") {
            throw std::runtime_error("Unsupported checkpoint architecture");
        }

        const auto count = gguf_get_n_tensors(file.get());
        std::ofstream dump;
        if (argc == 3) {
            dump.open(argv[2], std::ios::binary);
            if (!dump) {
                throw std::runtime_error("Cannot open the tensor dump");
            }
        }

        for (int64_t i = 0; i < count; ++i) {
            const std::string name = gguf_get_tensor_name(file.get(), i);
            const auto rank_key = gguf_find_key(file.get(), ("propainter.rank." + name).c_str());
            if (rank_key < 0 || gguf_get_kv_type(file.get(), rank_key) != GGUF_TYPE_UINT32) {
                throw std::runtime_error("Missing original rank for " + name);
            }
            const auto rank = gguf_get_val_u32(file.get(), rank_key);
            const auto dtype = gguf_get_tensor_type(file.get(), i);
            if (rank > 5 || (dtype != GGML_TYPE_F32 && dtype != GGML_TYPE_I64)) {
                throw std::runtime_error("Unsupported tensor metadata for " + name);
            }
            std::cout << name << ' ' << ggml_type_name(dtype) << ' ' << rank;
            if (rank > 0) {
                const auto shape_key = gguf_find_key(file.get(), ("propainter.shape." + name).c_str());
                if (shape_key < 0 || gguf_get_kv_type(file.get(), shape_key) != GGUF_TYPE_ARRAY ||
                    gguf_get_arr_type(file.get(), shape_key) != GGUF_TYPE_INT32 || gguf_get_arr_n(file.get(), shape_key) != rank) {
                    throw std::runtime_error("Invalid original shape for " + name);
                }
                const auto * shape = static_cast<const int32_t *>(gguf_get_arr_data(file.get(), shape_key));
                for (size_t dimension = 0; dimension < rank; ++dimension) {
                    std::cout << ' ' << shape[dimension];
                }
            }
            std::cout << '\n';
            if (storage) {
                auto * tensor = ggml_get_tensor(tensors.get(), name.c_str());
                if (!tensor || !tensor->data) {
                    throw std::runtime_error("Missing tensor data for " + name);
                }
                dump.write(static_cast<const char *>(tensor->data), ggml_nbytes(tensor));
                if (!dump) {
                    throw std::runtime_error("Could not write the tensor dump");
                }
            }
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
