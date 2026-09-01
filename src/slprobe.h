#pragma once

#include <string>

// Ask Streamline itself whether DLSS Neural Rendering (feature 1004) can run on this
// machine, driving sl.interposer.dll exactly as a game would: slInit, then
// slIsFeatureSupported / slGetFeatureRequirements. Streamline carries its own NGX core
// (rel_310_8) and its own snippet loader, so this path does not touch the driver's
// _nvngx.dll feature gate that route A hits. Returns 0 if the feature reports supported.
int ProbeStreamlineNR(const std::string& pluginDir, unsigned featureId, bool verbose);
