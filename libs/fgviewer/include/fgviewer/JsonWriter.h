/*
 * Copyright (C) 2024 The Android Open Source Project
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

#ifndef FGVIEWER_JSONWRITER_H
#define FGVIEWER_JSONWRITER_H

#include <utils/CString.h>

namespace filament::fgviewer {

struct FrameGraphInfo;

// This class generates portions of JSON messages that are sent to the web client.
// Note that some portions of these JSON strings are generated by directly in DebugServer,
// as well as CommonWriter.
class JsonWriter {
public:

    // Retrieves the most recently generated string.
    const char* getJsonString() const;
    size_t getJsonSize() const;

    // Generates a JSON string describing the given FrameGraphInfo.
    bool writeFrameGraphInfo(const FrameGraphInfo& frameGraph);

private:
    utils::CString mJsonString;
};

} // namespace filament::fgviewer

#endif  // FGVIEWER_JSONWRITER_H
