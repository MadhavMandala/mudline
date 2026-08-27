"""GLSL for the viewport.

Two programs. A lit shader for solids, and an unlit one for lines -- grid,
axes, trajectories. Keeping lines out of the lighting path is what stops a
grid from being shaded into invisibility at grazing angles.

There is no distance fog here, deliberately. The old shader faded everything to
the clear colour past a fixed 400 m, which made any view wider than that render
as a blank frame. Depth cueing is done instead by fading the *grid* with
distance, which is a scale cue rather than a way to lose the model.
"""

SOLID_VERT = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 v_world;
out vec3 v_normal;

void main() {
    vec4 world = model * vec4(in_position, 1.0);
    v_world = world.xyz;
    // The display transform is rigid -- a rotation and a translation, with no
    // scale or shear -- so its upper 3x3 *is* the normal matrix. Deriving it
    // here rather than passing a separate mat3 uniform removes a uniform whose
    // byte layout moderngl and the driver need not agree on.
    v_normal = normalize(mat3(model) * in_normal);
    gl_Position = projection * view * world;
}
"""

SOLID_FRAG = """
#version 330 core
in vec3 v_world;
in vec3 v_normal;

uniform vec3 light_dir;
uniform vec3 view_pos;
uniform vec3 base_color;
uniform float selected;
uniform float alpha;
//: 0 for a matte composite, 1 for a polished metal.
//
// Colour on its own is not enough to tell materials apart: a grey tube and a
// black one under one fixed highlight both read as painted plastic. What says
// "metal" is a tight bright specular, and what says "laminate" is a broad dull
// one, so the material drives the highlight as well as the hue.
uniform float sheen;
//: Half-section. xyz is the plane normal, w the offset; anything on the
// positive side is thrown away.
//
// Discarding fragments rather than cutting the solid: the geometry is
// untouched, so nothing downstream -- mass, export, meshing -- sees a
// half-vehicle, and the cut costs one dot product per fragment instead of a
// boolean per part. The shader is already two-sided, so the shells light
// correctly from the inside once opened up.
uniform vec4 cut_plane;
uniform float cut_enabled;

out vec4 frag_color;

void main() {
    if (cut_enabled > 0.5 && dot(v_world, cut_plane.xyz) > cut_plane.w) discard;

    vec3 N = normalize(v_normal);
    vec3 V = normalize(view_pos - v_world);
    // Two-sided: a hollow shell shows its inner faces, and lighting them from
    // behind leaves the interior a flat black hole.
    if (dot(N, V) < 0.0) N = -N;

    vec3 L = normalize(light_dir);
    vec3 H = normalize(L + V);

    float shininess    = mix(10.0, 110.0, sheen);
    float spec_weight  = mix(0.05, 0.55, sheen);
    // A metal tints its highlight toward its own colour; a dielectric's stays
    // white. Interpolating between the two is the cheapest thing that reads as
    // the difference between anodised aluminium and painted glass.
    vec3  spec_tint    = mix(vec3(1.0), base_color, sheen * 0.7);

    float ambient  = 0.34;
    float diffuse  = max(dot(N, L), 0.0) * mix(0.70, 0.50, sheen);
    float specular = pow(max(dot(N, H), 0.0), shininess) * spec_weight;
    float rim      = pow(1.0 - max(dot(V, N), 0.0), 3.0) * 0.28;

    vec3 color = mix(base_color, vec3(1.0, 0.62, 0.18), selected * 0.65);
    vec3 lit = (ambient + diffuse) * color
             + specular * spec_tint
             + rim * vec3(0.42, 0.58, 0.85);

    frag_color = vec4(lit, alpha);
}
"""

LINE_VERT = """
#version 330 core
in vec3 in_position;
in vec3 in_color;

uniform mat4 view;
uniform mat4 projection;

out vec3 v_color;
out vec3 v_world;

void main() {
    v_color = in_color;
    v_world = in_position;
    gl_Position = projection * view * vec4(in_position, 1.0);
}
"""

LINE_FRAG = """
#version 330 core
in vec3 v_color;
in vec3 v_world;

uniform vec3 view_pos;
uniform float fade_start;
uniform float fade_end;
uniform vec3 background;

out vec4 frag_color;

void main() {
    // Grid lines fade with distance so the floor reads as receding rather than
    // turning into moire. Solids are untouched by this.
    float d = length(view_pos - v_world);
    float t = clamp((d - fade_start) / max(fade_end - fade_start, 1e-6), 0.0, 1.0);
    frag_color = vec4(mix(v_color, background, t * 0.92), 1.0);
}
"""
