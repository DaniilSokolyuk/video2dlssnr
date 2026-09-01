#pragma once
// Shared POD passed from the host to the forwarder. Same layout both sides.
// These are the model's own parameters, latched when the NR feature is created.
struct NrModelParams {
    unsigned preset;         // DLSSNR.Hint.Render.Preset: 0 Default, 1/2/3 Preset 1..3
    float    intensity;      // DLSSNR.Intensity: overall detail strength
    unsigned style;          // DLSSNR.Style: 0 Default, 1 Natural, 2 Cinematic
    float    localStructure; // DLSSNR.LocalStructureStrength
    float    localTone;      // DLSSNR.LocalToneStrength
    float    skinStructure;  // DLSSNR.SkinStructureStrength (<0 = leave at model default)
    float    globalTone;     // DLSSNR.GlobalToneStrength    (<0 = leave at model default)
    unsigned autoMask;       // DLSSNR.UseAutoMask: 0/1
    unsigned uiCorrection;   // DLSSNR.UICorrection: 0/1
};
