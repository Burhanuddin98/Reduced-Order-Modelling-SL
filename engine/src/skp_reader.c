/**
 * skp_reader.c — Read SketchUp .skp files and export geometry
 *
 * Uses the official SketchUp C SDK to extract:
 * - Vertices and faces (triangulated)
 * - Material names per face
 * - Bounding box
 *
 * Exports to simple OBJ format for downstream processing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <SketchUpAPI/initialize.h>
#include <SketchUpAPI/model/model.h>
#include <SketchUpAPI/model/entities.h>
#include <SketchUpAPI/model/face.h>
#include <SketchUpAPI/model/edge.h>
#include <SketchUpAPI/model/vertex.h>
#include <SketchUpAPI/model/loop.h>
#include <SketchUpAPI/model/mesh_helper.h>
#include <SketchUpAPI/model/material.h>
#include <SketchUpAPI/model/component_definition.h>
#include <SketchUpAPI/model/component_instance.h>
#include <SketchUpAPI/model/group.h>
#include <SketchUpAPI/geometry/point3d.h>
#include <SketchUpAPI/unicodestring.h>

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

/* ================================================================
 * Helper: get material name from a face
 * ================================================================ */
static void get_face_material(SUFaceRef face, char* mat_name, int max_len) {
    SUMaterialRef material = SU_INVALID;
    SUFaceGetFrontMaterial(face, &material);

    if (SUIsInvalid(material)) {
        SUFaceGetBackMaterial(face, &material);
    }

    if (SUIsValid(material)) {
        SUStringRef name = SU_INVALID;
        SUStringCreate(&name);
        SUMaterialGetName(material, &name);

        size_t len = 0;
        SUStringGetUTF8Length(name, &len);
        if (len > 0 && len < (size_t)max_len) {
            SUStringGetUTF8(name, max_len, mat_name, &len);
        } else {
            strncpy(mat_name, "default", max_len);
        }
        SUStringRelease(&name);
    } else {
        strncpy(mat_name, "default", max_len);
    }
}

/* ================================================================
 * Process entities (recursive for groups/components)
 * ================================================================ */
static int process_entities(SUEntitiesRef entities, FILE* obj_fp,
                            FILE* mtl_fp, int* vertex_offset) {
    /* Get faces */
    size_t n_faces = 0;
    SUEntitiesGetNumFaces(entities, &n_faces);

    if (n_faces > 0) {
        SUFaceRef* faces = (SUFaceRef*)malloc(n_faces * sizeof(SUFaceRef));
        SUEntitiesGetFaces(entities, n_faces, faces, &n_faces);

        for (size_t fi = 0; fi < n_faces; fi++) {
            /* Get material */
            char mat_name[256] = "default";
            get_face_material(faces[fi], mat_name, 256);
            fprintf(obj_fp, "usemtl %s\n", mat_name);

            /* Triangulate the face using mesh helper */
            SUMeshHelperRef mesh = SU_INVALID;
            SUMeshHelperCreate(&mesh, faces[fi]);

            size_t n_verts = 0, n_tris = 0;
            SUMeshHelperGetNumVertices(mesh, &n_verts);
            SUMeshHelperGetNumTriangles(mesh, &n_tris);

            if (n_verts > 0 && n_tris > 0) {
                /* Get vertices */
                SUPoint3D* points = (SUPoint3D*)malloc(n_verts * sizeof(SUPoint3D));
                SUMeshHelperGetVertices(mesh, n_verts, points, &n_verts);

                /* Write vertices (convert from inches to meters) */
                for (size_t vi = 0; vi < n_verts; vi++) {
                    fprintf(obj_fp, "v %.6f %.6f %.6f\n",
                            points[vi].x * 0.0254,
                            points[vi].y * 0.0254,
                            points[vi].z * 0.0254);
                }

                /* Get triangle indices */
                size_t* indices = (size_t*)malloc(n_tris * 3 * sizeof(size_t));
                SUMeshHelperGetVertexIndices(mesh, n_tris * 3, indices, &n_tris);

                /* Write faces (1-indexed in OBJ) */
                for (size_t ti = 0; ti < n_tris; ti++) {
                    fprintf(obj_fp, "f %d %d %d\n",
                            (int)(indices[ti*3+0] + 1 + *vertex_offset),
                            (int)(indices[ti*3+1] + 1 + *vertex_offset),
                            (int)(indices[ti*3+2] + 1 + *vertex_offset));
                }

                *vertex_offset += (int)n_verts;

                free(points);
                free(indices);
            }

            SUMeshHelperRelease(&mesh);
        }
        free(faces);
    }

    /* Recurse into groups */
    size_t n_groups = 0;
    SUEntitiesGetNumGroups(entities, &n_groups);
    if (n_groups > 0) {
        SUGroupRef* groups = (SUGroupRef*)malloc(n_groups * sizeof(SUGroupRef));
        SUEntitiesGetGroups(entities, n_groups, groups, &n_groups);
        for (size_t gi = 0; gi < n_groups; gi++) {
            SUEntitiesRef group_ents = SU_INVALID;
            SUGroupGetEntities(groups[gi], &group_ents);
            process_entities(group_ents, obj_fp, mtl_fp, vertex_offset);
        }
        free(groups);
    }

    /* Recurse into component instances */
    size_t n_instances = 0;
    SUEntitiesGetNumInstances(entities, &n_instances);
    if (n_instances > 0) {
        SUComponentInstanceRef* instances =
            (SUComponentInstanceRef*)malloc(n_instances * sizeof(SUComponentInstanceRef));
        SUEntitiesGetInstances(entities, n_instances, instances, &n_instances);
        for (size_t ii = 0; ii < n_instances; ii++) {
            SUComponentDefinitionRef def = SU_INVALID;
            SUComponentInstanceGetDefinition(instances[ii], &def);
            SUEntitiesRef def_ents = SU_INVALID;
            SUComponentDefinitionGetEntities(def, &def_ents);
            process_entities(def_ents, obj_fp, mtl_fp, vertex_offset);
        }
        free(instances);
    }

    return 0;
}

/* ================================================================
 * Main API: convert SKP to OBJ
 * ================================================================ */

EXPORT int skp_to_obj(const char* skp_path, const char* obj_path) {
    SUInitialize();

    SUModelRef model = SU_INVALID;
    SUResult res = SUModelCreateFromFile(&model, skp_path);
    if (res != SU_ERROR_NONE) {
        fprintf(stderr, "Failed to open SKP file: %s (error %d)\n", skp_path, res);
        SUTerminate();
        return -1;
    }

    FILE* obj_fp = fopen(obj_path, "w");
    if (!obj_fp) {
        fprintf(stderr, "Failed to open output file: %s\n", obj_path);
        SUModelRelease(&model);
        SUTerminate();
        return -2;
    }

    fprintf(obj_fp, "# Converted from %s by room_engine\n", skp_path);

    /* Get root entities */
    SUEntitiesRef entities = SU_INVALID;
    SUModelGetEntities(model, &entities);

    int vertex_offset = 0;
    process_entities(entities, obj_fp, NULL, &vertex_offset);

    fclose(obj_fp);
    SUModelRelease(&model);
    SUTerminate();

    fprintf(stderr, "SKP->OBJ: %s -> %s (%d vertices)\n",
            skp_path, obj_path, vertex_offset);
    return 0;
}

/* ================================================================
 * Query: get model info without full conversion
 * ================================================================ */

EXPORT int skp_info(const char* skp_path,
                     double* bbox_min, double* bbox_max,
                     int* n_faces_out) {
    SUInitialize();

    SUModelRef model = SU_INVALID;
    if (SUModelCreateFromFile(&model, skp_path) != SU_ERROR_NONE) {
        SUTerminate();
        return -1;
    }

    SUEntitiesRef entities = SU_INVALID;
    SUModelGetEntities(model, &entities);

    size_t n_faces = 0;
    SUEntitiesGetNumFaces(entities, &n_faces);
    *n_faces_out = (int)n_faces;

    /* TODO: compute bounding box by iterating vertices */
    bbox_min[0] = bbox_min[1] = bbox_min[2] = 0;
    bbox_max[0] = bbox_max[1] = bbox_max[2] = 0;

    SUModelRelease(&model);
    SUTerminate();
    return 0;
}
