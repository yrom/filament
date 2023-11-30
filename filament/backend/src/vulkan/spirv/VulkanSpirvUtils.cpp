/*
 * Copyright (C) 2023 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "VulkanSpirvUtils.h"

#include "../VulkanConstants.h"

#include <utils/Log.h>

#include <bluevk/BlueVK.h>
#include <spirv-tools/libspirv.hpp>

#include <regex>
#include <string_view>
#include <sstream>
#include <unordered_map>

namespace filament::backend {

using namespace spvtools;
using namespace bluevk;

namespace {

std::string getTransformedConstantStr(SpecConstantValue const& value) {
    if (std::holds_alternative<bool>(value)) {
        if (std::get<bool>(value)) {
            return "OpConstantTrue %bool";
        }
        return "OpConstantFalse %bool";
    } else if (std::holds_alternative<float>(value)) {
        float const fval = std::get<float>(value);
        return "OpConstant %float " + std::to_string(fval);
    } else {
        int32_t const ival = std::get<int32_t>(value);
        return "OpConstant %int " + std::to_string(ival);
    }
}

} // anonymous namespace

// TODO: Directly modifying the binary format will be more performant.
void workaroundSpecConstant(Program::ShaderBlob const& blob,
        utils::FixedCapacityVector<Program::SpecializationConstant> const& specConstants,
        std::vector<uint32_t>& output) {
    using IdToValueMap = std::unordered_map<uint8_t, SpecConstantValue>;
    using VarToIdMap = std::unordered_map<std::string, uint8_t>;

    IdToValueMap idToValue;
    for (auto spec: specConstants) {
        idToValue[spec.id] = spec.value;
    }

    SpirvTools const tools(SPV_ENV_UNIVERSAL_1_3);
    std::string disassembly;
    UTILS_UNUSED_IN_RELEASE bool const result =
            tools.Disassemble((uint32_t*) blob.data(), blob.size() / 4, &disassembly);

    assert_invariant(result && "Cannot disassemble shader blob");

    // The first step is to remove the `OpDecorate` lines for the spec consts. During this
    // pass, we also map the variable name to a spec const ID. Later lines will assign values to the
    // variable and we need to use the specialization value associated with this ID.
    //
    // The second step is to change a line of the form `%var = OpSpecConstant` to
    // `%var = OpConstant`.

    std::stringstream ss(disassembly);
    std::regex const decorateRegex("OpDecorate (\\%[\\w]+) SpecId ([\\d]+)");
    std::regex const assignRegex("(\\%[\\w]+) = OpSpecConstant.+");

    VarToIdMap varToId;
    std::string transformedOutput;
    for (std::string line; std::getline(ss, line);) {
        std::smatch match;
        std::string transformedLine = line + '\n';
        if (std::regex_match(line, match, decorateRegex) && match.size() == 3) {
            varToId[match[1].str()] = std::stoi(match[2].str());
            transformedLine = "";
        } else if (line.find("OpSpecConstant") != std::string::npos) {
            UTILS_UNUSED_IN_RELEASE bool const matches =
                    std::regex_match(line, match, assignRegex) && match.size() == 2;
            assert_invariant(matches && "Falsed to match a spec const assignment");
            std::string const assignedVar = match[1].str();
            VarToIdMap::const_iterator idItr = varToId.find(assignedVar);
            IdToValueMap::const_iterator valItr;
            if (idItr != varToId.end() &&
                    (valItr = idToValue.find(idItr->second)) != idToValue.end()) {
                transformedLine =
                        assignedVar + " = " + getTransformedConstantStr(valItr->second) + '\n';
            }
        }
        transformedOutput += transformedLine;
    }

    tools.Assemble(transformedOutput, &output);
    assert_invariant(tools.Validate(output));
}

} // namespace filament::backend
