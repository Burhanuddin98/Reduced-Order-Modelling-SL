/**
 * ray_tracer.c — Fast stochastic ray tracer for room acoustics
 *
 * Traces N rays from source through triangle mesh, records energy
 * arriving near receiver. Returns a reflectogram (energy histogram).
 *
 * Vectorized ray-triangle intersection using Moller-Trumbore.
 * All triangles precomputed as edge vectors for fast testing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

/* ================================================================ */
/* Vector math                                                       */
/* ================================================================ */

typedef struct { double x, y, z; } Vec3;

static Vec3 vec3(double x, double y, double z) {
    Vec3 v = {x, y, z}; return v;
}
static Vec3 vsub(Vec3 a, Vec3 b) { return vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
static Vec3 vadd(Vec3 a, Vec3 b) { return vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
static Vec3 vscale(Vec3 a, double s) { return vec3(a.x*s, a.y*s, a.z*s); }
static double vdot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static Vec3 vcross(Vec3 a, Vec3 b) {
    return vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static double vlen(Vec3 a) { return sqrt(vdot(a, a)); }
static Vec3 vnorm(Vec3 a) { double l = vlen(a); return l > 1e-15 ? vscale(a, 1.0/l) : vec3(0,0,1); }

/* ================================================================ */
/* Triangle mesh                                                     */
/* ================================================================ */

typedef struct {
    int n_tris;
    Vec3 *v0, *e1, *e2;    /* precomputed: v0, edge1=v1-v0, edge2=v2-v0 */
    Vec3 *normals;
    double *alpha;          /* absorption per triangle */
    double *scatter;        /* scattering per triangle */
} TriMesh;

/* ================================================================ */
/* Ray-mesh intersection (brute force, fast for <10K triangles)      */
/* ================================================================ */

static int intersect_mesh(const TriMesh* mesh, Vec3 origin, Vec3 dir,
                           double* t_out, int* tri_out, Vec3* hit_out, Vec3* normal_out)
{
    double best_t = 1e30;
    int best_idx = -1;

    for (int i = 0; i < mesh->n_tris; i++) {
        Vec3 h = vcross(dir, mesh->e2[i]);
        double a = vdot(mesh->e1[i], h);
        if (fabs(a) < 1e-10) continue;

        double f = 1.0 / a;
        Vec3 s = vsub(origin, mesh->v0[i]);
        double u = f * vdot(s, h);
        if (u < 0.0 || u > 1.0) continue;

        Vec3 q = vcross(s, mesh->e1[i]);
        double v = f * vdot(dir, q);
        if (v < 0.0 || u + v > 1.0) continue;

        double t = f * vdot(mesh->e2[i], q);
        if (t > 1e-6 && t < best_t) {
            best_t = t;
            best_idx = i;
        }
    }

    if (best_idx < 0) return 0;

    *t_out = best_t;
    *tri_out = best_idx;
    *hit_out = vadd(origin, vscale(dir, best_t));
    *normal_out = mesh->normals[best_idx];
    return 1;
}

/* ================================================================ */
/* Simple LCG random number generator (deterministic, fast)          */
/* ================================================================ */

static unsigned int rng_state = 42;

static double rng_double(void) {
    rng_state = rng_state * 1103515245 + 12345;
    return (double)(rng_state & 0x7FFFFFFF) / 2147483647.0;
}

static Vec3 random_sphere(void) {
    double theta = acos(2.0 * rng_double() - 1.0);
    double phi = 2.0 * 3.14159265358979323846 * rng_double();
    return vec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
}

static Vec3 random_hemisphere(Vec3 normal) {
    /* Cosine-weighted hemisphere sampling */
    double u1 = rng_double();
    double u2 = rng_double();
    double cos_t = sqrt(u1);
    double sin_t = sqrt(1.0 - u1);
    double phi = 2.0 * 3.14159265358979323846 * u2;

    /* Build local frame */
    Vec3 tangent, bitangent;
    if (fabs(normal.x) < 0.9) {
        tangent = vnorm(vcross(normal, vec3(1, 0, 0)));
    } else {
        tangent = vnorm(vcross(normal, vec3(0, 1, 0)));
    }
    bitangent = vcross(normal, tangent);

    return vnorm(vadd(vadd(vscale(normal, cos_t),
                            vscale(tangent, sin_t * cos(phi))),
                       vscale(bitangent, sin_t * sin(phi))));
}

/* ================================================================ */
/* Main ray tracing function                                         */
/* ================================================================ */

EXPORT int ray_trace(
    /* Mesh */
    int n_tris,
    const double* vertices,     /* 3 * n_verts doubles (x,y,z interleaved) */
    const int* triangles,       /* 3 * n_tris ints (v0,v1,v2 per triangle) */
    const double* tri_alpha,    /* n_tris: absorption per triangle */
    const double* tri_scatter,  /* n_tris: scattering per triangle */
    /* Source & receiver */
    double src_x, double src_y, double src_z,
    double rec_x, double rec_y, double rec_z,
    double capture_radius,
    /* Ray params */
    int n_rays, int max_bounces,
    /* Output */
    double speed_of_sound, int sample_rate, double duration,
    double* reflectogram,       /* output: n_bins doubles */
    int* n_bins_out)
{
    int n_bins = (int)(duration * sample_rate);
    *n_bins_out = n_bins;
    memset(reflectogram, 0, n_bins * sizeof(double));

    /* Build triangle mesh */
    TriMesh mesh;
    mesh.n_tris = n_tris;
    mesh.v0 = (Vec3*)malloc(n_tris * sizeof(Vec3));
    mesh.e1 = (Vec3*)malloc(n_tris * sizeof(Vec3));
    mesh.e2 = (Vec3*)malloc(n_tris * sizeof(Vec3));
    mesh.normals = (Vec3*)malloc(n_tris * sizeof(Vec3));
    mesh.alpha = (double*)malloc(n_tris * sizeof(double));
    mesh.scatter = (double*)malloc(n_tris * sizeof(double));

    for (int i = 0; i < n_tris; i++) {
        int i0 = triangles[3*i+0];
        int i1 = triangles[3*i+1];
        int i2 = triangles[3*i+2];
        Vec3 va = vec3(vertices[3*i0], vertices[3*i0+1], vertices[3*i0+2]);
        Vec3 vb = vec3(vertices[3*i1], vertices[3*i1+1], vertices[3*i1+2]);
        Vec3 vc = vec3(vertices[3*i2], vertices[3*i2+1], vertices[3*i2+2]);
        mesh.v0[i] = va;
        mesh.e1[i] = vsub(vb, va);
        mesh.e2[i] = vsub(vc, va);
        mesh.normals[i] = vnorm(vcross(mesh.e1[i], mesh.e2[i]));
        mesh.alpha[i] = tri_alpha[i];
        mesh.scatter[i] = tri_scatter[i];
    }

    Vec3 src = vec3(src_x, src_y, src_z);
    Vec3 rec = vec3(rec_x, rec_y, rec_z);
    double cap_r2 = capture_radius * capture_radius;

    /* Direct sound */
    Vec3 direct = vsub(rec, src);
    double d_direct = vlen(direct);
    int n_direct = (int)(d_direct / speed_of_sound * sample_rate);
    if (n_direct >= 0 && n_direct < n_bins) {
        reflectogram[n_direct] += 1.0 / (4.0 * 3.14159265 * d_direct * d_direct);
    }

    /* Trace rays */
    rng_state = 42;
    for (int ray = 0; ray < n_rays; ray++) {
        Vec3 dir = random_sphere();
        Vec3 origin = src;
        double energy = 1.0;
        double total_dist = 0.0;

        for (int bounce = 0; bounce < max_bounces; bounce++) {
            double t_hit;
            int tri_idx;
            Vec3 hit, normal;

            if (!intersect_mesh(&mesh, origin, dir, &t_hit, &tri_idx, &hit, &normal))
                break;

            total_dist += t_hit;

            /* Absorb */
            energy *= (1.0 - mesh.alpha[tri_idx]);
            if (energy < 1e-8) break;

            /* Check receiver proximity along ray path */
            Vec3 ray_to_rec = vsub(rec, origin);
            double proj = vdot(ray_to_rec, dir);
            if (proj > 0 && proj < t_hit) {
                Vec3 closest = vadd(origin, vscale(dir, proj));
                Vec3 diff = vsub(closest, rec);
                double dist2 = vdot(diff, diff);
                if (dist2 < cap_r2) {
                    double arrival = total_dist / speed_of_sound;
                    int bin = (int)(arrival * sample_rate);
                    if (bin >= 0 && bin < n_bins) {
                        double dist = sqrt(dist2);
                        double weight = energy / (4.0 * 3.14159265 * (dist + 0.01) * (dist + 0.01));
                        weight *= cap_r2;
                        reflectogram[bin] += weight / n_rays;
                    }
                }
            }

            /* Reflect */
            if (rng_double() < mesh.scatter[tri_idx]) {
                dir = random_hemisphere(normal);
            } else {
                /* Specular */
                dir = vsub(dir, vscale(normal, 2.0 * vdot(dir, normal)));
                dir = vnorm(dir);
            }
            origin = vadd(hit, vscale(dir, 1e-6));
        }
    }

    /* Cleanup */
    free(mesh.v0); free(mesh.e1); free(mesh.e2);
    free(mesh.normals); free(mesh.alpha); free(mesh.scatter);

    {
        int count = 0;
        for (int i = 0; i < n_bins; i++) if (reflectogram[i] > 0) count++;
        fprintf(stderr, "ray_trace: %d rays, %d tris, %d bins with energy\n",
                n_rays, n_tris, count);
    }

    return 0;
}
