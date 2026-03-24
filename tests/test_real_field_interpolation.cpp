#include <cstdio>

#include "mathtools/real_field_3d.hpp"
#include "tests/test_utils.hpp"

static const int N = 14;  // grid size per axis; large enough to avoid boundary
                           // clipping for any stencil used in these tests

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void setupMesh(RealField3D &mesh) {
    for (int ax = 0; ax < 3; ++ax) {
        mesh.ngrid[ax] = N;
        mesh.length[ax] = static_cast<float>(N);
        mesh.xyz0[ax] = 0.0f;
    }
    mesh.construct(/*inplace=*/false);
    mesh.zero_field_x();
}

// Sum every valid real-space cell (respects the ngrid_z padding stride).
static double sumField(const RealField3D &mesh) {
    double total = 0;
    for (size_t ij = 0; ij < mesh.ngrid_xy; ++ij)
        for (int iz = 0; iz < mesh.ngrid[2]; ++iz)
            total += mesh.field_x[iz + mesh.ngrid_z * ij];
    return total;
}

// ---------------------------------------------------------------------------
// Test 1 & 2: mass conservation
// ---------------------------------------------------------------------------
// TSC (and CIC) weights form a partition of unity: they sum to 1 over the
// stencil.  Depositing a value `val` via reverseInterpolate must therefore
// leave the total grid sum equal to `val`.

int test_tsc_mass_conservation() {
    RealField3D mesh(/*cic=*/false);
    setupMesh(mesh);

    const double val = 3.14159;
    float coord[3] = {5.3f, 5.3f, 5.3f};
    mesh.reverseInterpolateTSC(coord, val);

    double total = sumField(mesh);
    if (isClose(total, val, 1e-7, 0))
        return 0;

    fprintf(stderr, "ERROR test_tsc_mass_conservation\n");
    printValues(val, total);
    return 1;
}

int test_cic_mass_conservation() {
    RealField3D mesh(/*cic=*/true);
    setupMesh(mesh);

    const double val = 2.71828;
    float coord[3] = {5.3f, 5.3f, 5.3f};
    mesh.reverseInterpolateCIC(coord, val);

    double total = sumField(mesh);
    if (isClose(total, val, 1e-7, 0))
        return 0;

    fprintf(stderr, "ERROR test_cic_mass_conservation\n");
    printValues(val, total);
    return 1;
}

// ---------------------------------------------------------------------------
// Test 3: TSC center-cell weight
// ---------------------------------------------------------------------------
// When a particle sits exactly at a cell center (fractional offset d = 0),
// the tscWeights function gives:
//   wm = 0.5 * (0.5)^2 = 0.125,  w0 = 0.75,  wp = 0.5 * (0.5)^2 = 0.125
// The center cell therefore accumulates w0^3 = 0.421875 of the deposited value.

int test_tsc_center_cell_weight() {
    RealField3D mesh(/*cic=*/false);
    setupMesh(mesh);

    const double val = 1.0;
    // dx = 1.0, so coord = 5.0 -> d = 5.0 -> n = roundf(5.0) = 5,
    // fractional offset = 0.
    float coord[3] = {5.0f, 5.0f, 5.0f};
    mesh.reverseInterpolateTSC(coord, val);

    const double expected_center_fraction = 0.75 * 0.75 * 0.75;  // 0.421875
    size_t center_idx = 5 + mesh.ngrid_z * (5 + mesh.ngrid[1] * 5);
    double center_val = mesh.field_x[center_idx];

    if (isClose(center_val, expected_center_fraction, 1e-8, 0))
        return 0;

    fprintf(stderr, "ERROR test_tsc_center_cell_weight\n");
    printValues(expected_center_fraction, center_val);
    return 1;
}

// ---------------------------------------------------------------------------
// Test 4: CIC corner weight
// ---------------------------------------------------------------------------
// coord / dx lands exactly on cell index 5 (fractional offset is 0).
// CIC computes w[axis][1] = 0 (right weight) and w[axis][0] = 1 (left weight),
// so all of `val` goes to cell (5, 5, 5).

int test_cic_corner_weight() {
    RealField3D mesh(/*cic=*/true);
    setupMesh(mesh);

    const double val = 7.5;
    float coord[3] = {5.0f, 5.0f, 5.0f};
    mesh.reverseInterpolateCIC(coord, val);

    size_t corner_idx = 5 + mesh.ngrid_z * (5 + mesh.ngrid[1] * 5);
    double corner_val = mesh.field_x[corner_idx];

    if (isClose(corner_val, val, 1e-7, 0))
        return 0;

    fprintf(stderr, "ERROR test_cic_corner_weight\n");
    printValues(val, corner_val);
    return 1;
}

// ---------------------------------------------------------------------------
// Tests 5 & 6: adjoint (forward–reverse consistency)
// ---------------------------------------------------------------------------
// For any field F and any particle value v deposited at coordinate c:
//
//   <F, reverseInterpolate(c, v)> = v * forwardInterpolate(F, c)
//
// This is the defining adjoint relation between the two operations and tests
// both simultaneously.  We verify it with a deterministic non-trivial field F.

static int test_adjoint(bool cic, const char *name) {
    RealField3D mesh_a(cic), mesh_b(cic);

    setupMesh(mesh_a);

    // mesh_b is a second mesh with identical grid configuration.
    mesh_b.copy(mesh_a);
    mesh_b.construct(/*inplace=*/false);
    mesh_b.zero_field_x();

    // Fill mesh_a with a deterministic non-constant pattern.
    for (int ix = 0; ix < N; ++ix) {
        for (int iy = 0; iy < N; ++iy) {
            for (int iz = 0; iz < N; ++iz) {
                size_t idx = iz + mesh_a.ngrid_z * (iy + N * ix);
                mesh_a.field_x[idx] = 0.7 * ix - 0.5 * iy + 0.3 * iz + 1.0;
            }
        }
    }

    const double val = 1.0;
    float coord[3] = {5.7f, 4.2f, 6.9f};

    // Deposit into mesh_b and forward-interpolate mesh_a at the same coord.
    double forward_val;
    if (cic) {
        mesh_b.reverseInterpolateCIC(coord, val);
        forward_val = mesh_a.forwardInterpolateCIC(coord);
    } else {
        mesh_b.reverseInterpolateTSC(coord, val);
        forward_val = mesh_a.forwardInterpolateTSC(coord);
    }

    double inner = mesh_a.dot(mesh_b);  // <F, R(c, val)>

    if (isClose(inner, val * forward_val, 1e-10, 1e-12))
        return 0;

    fprintf(stderr, "ERROR %s: inner=%.15g  val*forward=%.15g\n",
            name, inner, val * forward_val);
    return 1;
}

int test_tsc_adjoint() { return test_adjoint(false, "test_tsc_adjoint"); }
int test_cic_adjoint() { return test_adjoint(true,  "test_cic_adjoint"); }

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    int failures = 0;

    failures += test_tsc_mass_conservation();
    failures += test_cic_mass_conservation();
    failures += test_tsc_center_cell_weight();
    failures += test_cic_corner_weight();
    failures += test_tsc_adjoint();
    failures += test_cic_adjoint();

    if (failures == 0)
        printf("All real_field_3d interpolation tests passed.\n");
    else
        fprintf(stderr, "%d interpolation test(s) FAILED.\n", failures);

    return failures;
}
