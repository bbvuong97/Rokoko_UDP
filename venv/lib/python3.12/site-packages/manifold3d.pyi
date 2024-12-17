from collections.abc import Callable, Sequence
import enum
from typing import Annotated, overload

from numpy.typing import ArrayLike


import numpy as np
from typing import Literal, TypeVar, Union

N = TypeVar('N', bound=np.generic)
DoubleNx2 = np.ndarray[tuple[N, Literal[2]], np.dtype[np.double]]
Doublex2 = Union[np.ndarray[tuple[Literal[2]], np.dtype[np.double]], tuple[float, float], list[float]]
Doublex3 = Union[np.ndarray[tuple[Literal[3]], np.dtype[np.double]], tuple[float, float, float], list[float]]
Double2x3 = np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.double]]
Double3x4 = np.ndarray[tuple[Literal[3], Literal[4]], np.dtype[np.double]]
DoubleNx3 = np.ndarray[tuple[N, Literal[3]], np.dtype[np.double]]
Intx3 = Union[np.ndarray[tuple[Literal[3]], np.dtype[np.integer]], tuple[int, int, int], list[int]]
IntNx3 = np.ndarray[tuple[N, Literal[3]], np.dtype[np.integer]]

class CrossSection:
    """
    Two-dimensional cross sections guaranteed to be without self-intersections, or overlaps between polygons (from construction onwards). This class makes use of the [Clipper2](http://www.angusj.com/clipper2/Docs/Overview.htm) library for polygon clipping (boolean) and offsetting operations.
    """

    @overload
    def __init__(self) -> None:
        """
        The default constructor is an empty cross-section (containing no contours).
        """

    @overload
    def __init__(self, contours: Sequence[DoubleNx2], fillrule: FillRule = FillRule.Positive) -> None:
        """
        Create a 2d cross-section from a set of contours (complex polygons). A
        boolean union operation (with Positive filling rule by default) is
        performed to combine overlapping polygons and ensure the resulting
        CrossSection is free of intersections.
        :param contours: A set of closed paths describing zero or more complex
        polygons.
        :param fillrule: The filling rule used to interpret polygon sub-regions in
        contours.
        """

    def area(self) -> float:
        """
        Return the total area covered by complex polygons making up the
        CrossSection.
        """

    def num_vert(self) -> int:
        """Return the number of vertices in the CrossSection."""

    def num_contour(self) -> int:
        """
        Return the number of contours (both outer and inner paths) in the
        CrossSection.
        """

    def is_empty(self) -> bool:
        """Does the CrossSection contain any contours?"""

    def bounds(self) -> tuple:
        """
        Return bounding box of CrossSection as tuple(min_x, min_y, max_x, max_y)
        """

    def translate(self, v: Doublex2) -> CrossSection:
        """
        Move this CrossSection in space. This operation can be chained. Transforms
        are combined and applied lazily.
        :param v: The vector to add to every vertex.
        """

    def rotate(self, degrees: float) -> CrossSection:
        """
        Applies a (Z-axis) rotation to the CrossSection, in degrees. This operation
        can be chained. Transforms are combined and applied lazily.
        :param degrees: degrees about the Z-axis to rotate.
        """

    @overload
    def scale(self, scale: Doublex2) -> CrossSection:
        """
        Scale this CrossSection in space. This operation can be chained. Transforms
        are combined and applied lazily.
        :param scale: The vector to multiply every vertex by per component.
        """

    @overload
    def scale(self, s: float) -> None:
        """
        Scale this CrossSection in space. This operation can be chained. Transforms are combined and applied lazily.

        :param s: The scalar to multiply every vertex by per component.
        """

    def mirror(self, ax: Doublex2) -> CrossSection:
        """
        Mirror this CrossSection over the arbitrary axis described by the unit form
        of the given vector. If the length of the vector is zero, an empty
        CrossSection is returned. This operation can be chained. Transforms are
        combined and applied lazily.
        :param ax: the axis to be mirrored over
        """

    def transform(self, m: Double2x3) -> CrossSection:
        """
        Transform this CrossSection in space. The first two columns form a 2x2
        matrix transform and the last is a translation vector. This operation can
        be chained. Transforms are combined and applied lazily.
        :param m: The affine transform matrix to apply to all the vertices.
        """

    def warp(self, warp_func: Callable[[Doublex2], Doublex2]) -> CrossSection:
        """
        Move the vertices of this CrossSection (creating a new one) according to
        any arbitrary input function, followed by a union operation (with a
        Positive fill rule) that ensures any introduced intersections are not
        included in the result.
        :param warp_func: A function that takes the original vertex position and
        return the new position.
        """

    @overload
    def warp_batch(self, warp_func: Callable[[DoubleNx2], None]) -> CrossSection:
        """
        Same as CrossSection::warp but calls warpFunc with
        an ndarray[n, 2] instead of processing only one vertex at a time.
        :param warp_func: A function that takes multiple vertex positions as an
        ndarray[n, 2] and returns the new vertex positions.
        """

    @overload
    def warp_batch(self, warp_func: Callable[[DoubleNx2], object]) -> CrossSection: ...

    def simplify(self, epsilon: float = 1e-06) -> CrossSection:
        """
        Remove vertices from the contours in this CrossSection that are less than
        the specified distance epsilon from an imaginary line that passes through
        its two adjacent vertices. Near duplicate vertices and collinear points
        will be removed at lower epsilons, with elimination of line segments
        becoming increasingly aggressive with larger epsilons.
        It is recommended to apply this function following Offset, in order to
        clean up any spurious tiny line segments introduced that do not improve
        quality in any meaningful way. This is particularly important if further
        offseting operations are to be performed, which would compound the issue.
        """

    def offset(self, delta: float, join_type: JoinType, miter_limit: float = 2.0, circular_segments: int = 0) -> CrossSection:
        """
        Inflate the contours in CrossSection by the specified delta, handling
        corners according to the given JoinType.
        :param delta: Positive deltas will cause the expansion of outlining contours
        to expand, and retraction of inner (hole) contours. Negative deltas will
        have the opposite effect.
        :param jointype: The join type specifying the treatment of contour joins
        (corners).
        :param miter_limit: The maximum distance in multiples of delta that vertices
        can be offset from their original positions with before squaring is
        applied, <B>when the join type is Miter</B> (default is 2, which is the
        minimum allowed). See the [Clipper2
        MiterLimit](http://www.angusj.com/clipper2/Docs/Units/Clipper.Offset/Classes/ClipperOffset/Properties/MiterLimit.htm)
        page for a visual example.
        :param circular_segments: Number of segments per 360 degrees of
        <B>JoinType::Round</B> corners (roughly, the number of vertices that
        will be added to each contour). Default is calculated by the static Quality
        defaults according to the radius.
        """

    def __add__(self, arg: CrossSection, /) -> CrossSection:
        """Compute the boolean union between two cross-sections."""

    def __sub__(self, arg: CrossSection, /) -> CrossSection:
        """
        Compute the boolean difference of a (clip) cross-section from another
        (subject).
        """

    def __xor__(self, arg: CrossSection, /) -> CrossSection:
        """Compute the boolean intersection between two cross-sections."""

    def hull(self) -> CrossSection:
        """Compute the convex hull of this cross-section."""

    @staticmethod
    def batch_hull(cross_sections: Sequence[CrossSection]) -> CrossSection:
        """
        Compute the convex hull enveloping a set of cross-sections.
        :param cross_sections: A vector of cross-sections over which to compute a
        convex hull.
        """

    @staticmethod
    def hull_points(pts: DoubleNx2) -> CrossSection:
        """
        Compute the convex hull of a set of points. If the given points are fewer
        than 3, an empty CrossSection will be returned.
        :param pts: A vector of 2-dimensional points over which to compute a convex
        hull.
        """

    def decompose(self) -> list[CrossSection]:
        """
        This operation returns a vector of CrossSections that are topologically
        disconnected, each containing one outline contour with zero or more
        holes.
        """

    @staticmethod
    def batch_boolean(cross_sections: Sequence[CrossSection], op: OpType) -> CrossSection:
        """
        Perform the given boolean operation on a list of CrossSections. In case of
        Subtract, all CrossSections in the tail are differenced from the head.
        """

    @staticmethod
    def compose(cross_sections: Sequence[CrossSection]) -> CrossSection:
        """
        Construct a CrossSection from a vector of other CrossSections (batch
        boolean union).
        """

    def to_polygons(self) -> list[DoubleNx2]:
        """Return the contours of this CrossSection as a Polygons."""

    def extrude(self, height: float, n_divisions: int = 0, twist_degrees: float = 0.0, scale_top: Doublex2 = (1.0, 1.0)) -> Manifold:
        """
        Constructs a manifold from a set of polygons by extruding them along the
        Z-axis.
        Note that high twistDegrees with small nDivisions may cause
        self-intersection. This is not checked here and it is up to the user to
        choose the correct parameters.
        :param cross_section: A set of non-overlapping polygons to extrude.
        :param height: Z-extent of extrusion.
        :param n_divisions: Number of extra copies of the crossSection to insert into
        the shape vertically; especially useful in combination with twistDegrees to
        avoid interpolation artifacts. Default is none.
        :param twist_degrees: Amount to twist the top crossSection relative to the
        bottom, interpolated linearly for the divisions in between.
        :param scale_top: Amount to scale the top (independently in X and Y). If the
        scale is {0, 0}, a pure cone is formed with only a single vertex at the top.
        Note that scale is applied after twist.
        Default {1, 1}.
        """

    def revolve(self, circular_segments: int = 0, revolve_degrees: float = 360.0) -> Manifold:
        """
        Constructs a manifold from a set of polygons by revolving this cross-section
        around its Y-axis and then setting this as the Z-axis of the resulting
        manifold. If the polygons cross the Y-axis, only the part on the positive X
        side is used. Geometrically valid input will result in geometrically valid
        output.
        :param cross_section: A set of non-overlapping polygons to revolve.
        :param circular_segments: Number of segments along its diameter. Default is
        calculated by the static Defaults.
        :param revolve_degrees: Number of degrees to revolve. Default is 360 degrees.
        """

    @staticmethod
    def square(size: Doublex2, center: bool = False) -> CrossSection:
        """
        Constructs a square with the given XY dimensions. By default it is
        positioned in the first quadrant, touching the origin. If any dimensions in
        size are negative, or if all are zero, an empty Manifold will be returned.
        :param size: The X, and Y dimensions of the square.
        :param center: Set to true to shift the center to the origin.
        """

    @staticmethod
    def circle(radius: float, circular_segments: int = 0) -> CrossSection:
        """
        Constructs a circle of a given radius.
        :param radius: Radius of the circle. Must be positive.
        :param circular_segments: Number of segments along its diameter. Default is
        calculated by the static Quality defaults according to the radius.
        """

class Error(enum.Enum):
    NoError = 0

    NonFiniteVertex = 1

    NotManifold = 2

    VertexOutOfBounds = 3

    PropertiesWrongLength = 4

    MissingPositionProperties = 5

    MergeVectorsDifferentLengths = 6

    MergeIndexOutOfBounds = 7

    TransformWrongLength = 8

    RunIndexWrongLength = 9

    FaceIDWrongLength = 10

    InvalidConstruction = 11

class FillRule(enum.Enum):
    EvenOdd = 0
    """Only odd numbered sub-regions are filled."""

    NonZero = 1
    """Only non-zero sub-regions are filled."""

    Positive = 2
    """Only sub-regions with winding counts > 0 are filled."""

    Negative = 3
    """Only sub-regions with winding counts < 0 are filled."""

class JoinType(enum.Enum):
    Square = 0
    """
    Squaring is applied uniformly at all joins where the internal join angle is less that 90 degrees. The squared edge will be at exactly the offset distance from the join vertex.
    """

    Round = 1
    """
    Rounding is applied to all joins that have convex external angles, and it maintains the exact offset distance from the join vertex.
    """

    Miter = 2
    """
    There's a necessary limit to mitered joins (to avoid narrow angled joins producing excessively long and narrow [spikes](http://www.angusj.com/clipper2/Docs/Units/Clipper.Offset/Classes/ClipperOffset/Properties/MiterLimit.htm)). So where mitered joins would exceed a given maximum miter distance (relative to the offset distance), these are 'squared' instead.
    """

class Manifold:
    @overload
    def __init__(self) -> None:
        """Construct an empty Manifold."""

    @overload
    def __init__(self, mesh: Mesh) -> None:
        """
        Convert a Mesh into a Manifold, retaining its properties and merging only
        the positions according to the merge vectors. Will return an empty Manifold
        and set an Error Status if the result is not an oriented 2-manifold. Will
        collapse degenerate triangles and unnecessary vertices.
        All fields are read, making this structure suitable for a lossless round-trip
        of data from GetMesh. For multi-material input, use ReserveIDs to set a
        unique originalID for each material, and sort the materials into triangle
        runs.
        :param mesh: The input Mesh.
        """

    @overload
    def __init__(self, mesh: Mesh64) -> None:
        """
        Convert a Mesh into a Manifold, retaining its properties and merging only
        the positions according to the merge vectors. Will return an empty Manifold
        and set an Error Status if the result is not an oriented 2-manifold. Will
        collapse degenerate triangles and unnecessary vertices.
        All fields are read, making this structure suitable for a lossless round-trip
        of data from GetMesh. For multi-material input, use ReserveIDs to set a
        unique originalID for each material, and sort the materials into triangle
        runs.
        :param mesh64: The input Mesh64.
        """

    def __add__(self, arg: Manifold, /) -> Manifold:
        """Shorthand for Boolean Union."""

    def __sub__(self, arg: Manifold, /) -> Manifold:
        """Shorthand for Boolean Difference."""

    def __xor__(self, arg: Manifold, /) -> Manifold:
        """Shorthand for Boolean Intersection."""

    def hull(self) -> Manifold:
        """Compute the convex hull of this manifold."""

    @staticmethod
    def batch_hull(manifolds: Sequence[Manifold]) -> Manifold:
        """
        Compute the convex hull enveloping a set of manifolds.
        :param manifolds: A vector of manifolds over which to compute a convex hull.
        """

    @staticmethod
    def hull_points(pts: DoubleNx3) -> Manifold:
        """
        Compute the convex hull of a set of points. If the given points are fewer
        than 4, or they are all coplanar, an empty Manifold will be returned.
        :param pts: A vector of 3-dimensional points over which to compute a convex
        hull.
        """

    def transform(self, m: Double3x4) -> Manifold:
        """
        Transform this Manifold in space. The first three columns form a 3x3 matrix
        transform and the last is a translation vector. This operation can be
        chained. Transforms are combined and applied lazily.
        :param m: The affine transform matrix to apply to all the vertices.
        """

    def translate(self, t: Doublex3) -> Manifold:
        """
        Move this Manifold in space. This operation can be chained. Transforms are
        combined and applied lazily.
        :param v: The vector to add to every vertex.
        """

    @overload
    def scale(self, v: Doublex3) -> Manifold:
        """
        Scale this Manifold in space. This operation can be chained. Transforms are
        combined and applied lazily.
        :param v: The vector to multiply every vertex by per component.
        """

    @overload
    def scale(self, s: float) -> None:
        """
        Scale this Manifold in space. This operation can be chained. Transforms are combined and applied lazily.

        :param s: The scalar to multiply every vertex by component.
        """

    def mirror(self, v: Doublex3) -> Manifold:
        """
        Mirror this Manifold over the plane described by the unit form of the given
        normal vector. If the length of the normal is zero, an empty Manifold is
        returned. This operation can be chained. Transforms are combined and applied
        lazily.
        :param normal: The normal vector of the plane to be mirrored over
        """

    def rotate(self, v: Doublex3) -> Manifold:
        """
        Applies an Euler angle rotation to the manifold, first about the X axis, then
        Y, then Z, in degrees. We use degrees so that we can minimize rounding error,
        and eliminate it completely for any multiples of 90 degrees. Additionally,
        more efficient code paths are used to update the manifold when the transforms
        only rotate by multiples of 90 degrees. This operation can be chained.
        Transforms are combined and applied lazily.
        :param v: [X, Y, Z] rotation in degrees.
        """

    def warp(self, warp_func: Callable[[Doublex3], Doublex3]) -> Manifold:
        """
        This function does not change the topology, but allows the vertices to be
        moved according to any arbitrary input function. It is easy to create a
        function that warps a geometrically valid object into one which overlaps, but
        that is not checked here, so it is up to the user to choose their function
        with discretion.
        :param warp_func: A function that takes the original vertex position and
        return the new position.
        """

    def warp_batch(self, warp_func: Callable[[DoubleNx3], object]) -> Manifold:
        """
        Same as Manifold::warp but calls warpFunc with with
        an ndarray[n, 3] instead of processing only one vertex at a time.
        :param warp_func: A function that takes multiple vertex positions as an
        ndarray[n, 3] and returns the new vertex positions. The result should have the
        same shape as the input.
        """

    def set_properties(self, new_num_prop: int, f: Callable[[Doublex3, Annotated[ArrayLike, dict(dtype='float64', order='C', writable=False)]], object]) -> Manifold:
        """
        Create a new copy of this manifold with updated vertex properties by
        supplying a function that takes the existing position and properties as
        input. You may specify any number of output properties, allowing creation and
        removal of channels. Note: undefined behavior will result if you read past
        the number of input properties or write past the number of output properties.
        If propFunc is a nullptr, this function will just set the channel to zeroes.
        :param num_prop: The new number of properties per vertex.
        :param prop_func: A function that modifies the properties of a given vertex.
        """

    def calculate_curvature(self, gaussian_idx: int, mean_idx: int) -> Manifold:
        """
        Curvature is the inverse of the radius of curvature, and signed such that
        positive is convex and negative is concave. There are two orthogonal
        principal curvatures at any point on a manifold, with one maximum and the
        other minimum. Gaussian curvature is their product, while mean
        curvature is their sum. This approximates them for every vertex and assigns
        them as vertex properties on the given channels.
        :param gaussian_idx: The property channel index in which to store the Gaussian
        curvature. An index < 0 will be ignored (stores nothing). The property set
        will be automatically expanded to include the channel index specified.
        :param mean_idx: The property channel index in which to store the mean
        curvature. An index < 0 will be ignored (stores nothing). The property set
        will be automatically expanded to include the channel index specified.
        """

    def min_gap(self, other: Manifold, search_length: float) -> float:
        """
        Returns the minimum gap between two manifolds.Returns a double between 0 and searchLength.
        """

    def calculate_normals(self, normal_idx: int, min_sharp_angle: float = 60) -> Manifold:
        """
        Fills in vertex properties for normal vectors, calculated from the mesh
        geometry. Flat faces composed of three or more triangles will remain flat.
        :param normal_idx: The property channel in which to store the X
        values of the normals. The X, Y, and Z channels will be sequential. The
        property set will be automatically expanded such that NumProp will be at
        least normalIdx + 3.
        :param min_sharp_angle: Any edges with angles greater than this value will
        remain sharp, getting different normal vector properties on each side of the
        edge. By default, no edges are sharp and all normals are shared. With a value
        of zero, the model is faceted and all normals match their triangle normals,
        but in this case it would be better not to calculate normals at all.
        """

    def smooth_by_normals(self, normal_idx: int) -> Manifold:
        """
        Smooths out the Manifold by filling in the halfedgeTangent vectors. The
        geometry will remain unchanged until Refine or RefineToLength is called to
        interpolate the surface. This version uses the supplied vertex normal
        properties to define the tangent vectors. Faces of two coplanar triangles
        will be marked as quads, while faces with three or more will be flat.
        :param normal_idx: The first property channel of the normals. NumProp must be
        at least normalIdx + 3. Any vertex where multiple normals exist and don't
        agree will result in a sharp edge.
        """

    def smooth_out(self, min_sharp_angle: float = 60, min_smoothness: float = 0) -> Manifold:
        """
        Smooths out the Manifold by filling in the halfedgeTangent vectors. The
        geometry will remain unchanged until Refine or RefineToLength is called to
        interpolate the surface. This version uses the geometry of the triangles and
        pseudo-normals to define the tangent vectors. Faces of two coplanar triangles
        will be marked as quads.
        :param min_sharp_angle: degrees, default 60. Any edges with angles greater than
        this value will remain sharp. The rest will be smoothed to G1 continuity,
        with the caveat that flat faces of three or more triangles will always remain
        flat. With a value of zero, the model is faceted, but in this case there is
        no point in smoothing.
        :param min_smoothness: range: 0 - 1, default 0. The smoothness applied to sharp
        angles. The default gives a hard edge, while values > 0 will give a small
        fillet on these sharp edges. A value of 1 is equivalent to a minSharpAngle of
        180 - all edges will be smooth.
        """

    def refine(self, n: int) -> Manifold:
        """
        Increase the density of the mesh by splitting every edge into n pieces. For
        instance, with n = 2, each triangle will be split into 4 triangles. Quads
        will ignore their interior triangle bisector. These will all be coplanar (and
        will not be immediately collapsed) unless the Mesh/Manifold has
        halfedgeTangents specified (e.g. from the Smooth() constructor), in which
        case the new vertices will be moved to the interpolated surface according to
        their barycentric coordinates.
        :param n: The number of pieces to split every edge into. Must be > 1.
        """

    def refine_to_length(self, length: float) -> Manifold:
        """
        Increase the density of the mesh by splitting each edge into pieces of
        roughly the input length. Interior verts are added to keep the rest of the
        triangulation edges also of roughly the same length. If halfedgeTangents are
        present (e.g. from the Smooth() constructor), the new vertices will be moved
        to the interpolated surface according to their barycentric coordinates. Quads
        will ignore their interior triangle bisector.
        :param length: The length that edges will be broken down to.
        """

    def refine_to_tolerance(self, tolerance: float) -> Manifold:
        """
        Increase the density of the mesh by splitting each edge into pieces such that
        any point on the resulting triangles is roughly within tolerance of the
        smoothly curved surface defined by the tangent vectors. This means tightly
        curving regions will be divided more finely than smoother regions. If
        halfedgeTangents are not present, the result will simply be a copy of the
        original. Quads will ignore their interior triangle bisector.
        :param tolerance: The desired maximum distance between the faceted mesh
        produced and the exact smoothly curving surface. All vertices are exactly on
        the surface, within rounding error.
        """

    def to_mesh(self, normal_idx: int = -1) -> Mesh:
        """
        The most complete output of this library, returning a Mesh that is designed
        to easily push into a renderer, including all interleaved vertex properties
        that may have been input. It also includes relations to all the input meshes
        that form a part of this result and the transforms applied to each.
        :param normal_idx: If the original Mesh inputs that formed this manifold had
        properties corresponding to normal vectors, you can specify the first of the
        three consecutive property channels forming the (x, y, z) normals, which will
        cause this output Mesh to automatically update these normals according to
        the applied transforms and front/back side. normalIdx + 3 must be <=
        numProp, and all original Meshs must use the same channels for their
        normals.
        """

    def to_mesh64(self, normal_idx: int = -1) -> Mesh64:
        """
        The most complete output of this library, returning a Mesh that is designed
        to easily push into a renderer, including all interleaved vertex properties
        that may have been input. It also includes relations to all the input meshes
        that form a part of this result and the transforms applied to each.
        :param normal_idx: If the original Mesh inputs that formed this manifold had
        properties corresponding to normal vectors, you can specify the first of the
        three consecutive property channels forming the (x, y, z) normals, which will
        cause this output Mesh to automatically update these normals according to
        the applied transforms and front/back side. normalIdx + 3 must be <=
        numProp, and all original Meshs must use the same channels for their
        normals.
        """

    def num_vert(self) -> int:
        """The number of vertices in the Manifold."""

    def num_edge(self) -> int:
        """The number of edges in the Manifold."""

    def num_tri(self) -> int:
        """The number of triangles in the Manifold."""

    def num_prop(self) -> int:
        """The number of properties per vertex in the Manifold."""

    def num_prop_vert(self) -> int:
        """
        The number of property vertices in the Manifold. This will always be >=
        NumVert, as some physical vertices may be duplicated to account for different
        properties on different neighboring triangles.
        """

    def genus(self) -> int:
        """
        The genus is a topological property of the manifold, representing the number
        of "handles". A sphere is 0, torus 1, etc. It is only meaningful for a single
        mesh, so it is best to call Decompose() first.
        """

    def volume(self) -> float:
        """
        Get the volume of the manifold
         This is clamped to zero for a given face if they are within the Epsilon().
        """

    def surface_area(self) -> float:
        """
        Get the surface area of the manifold
         This is clamped to zero for a given face if they are within the Epsilon().
        """

    def original_id(self) -> int:
        """
        If this mesh is an original, this returns its meshID that can be referenced
        by product manifolds' MeshRelation. If this manifold is a product, this
        returns -1.
        """

    def get_tolerance(self) -> float:
        """
        Returns the tolerance value of this Manifold. Triangles that are coplanar
        within tolerance tend to be merged and edges shorter than tolerance tend to
        be collapsed.
        """

    def set_tolerance(self, arg: float, /) -> Manifold:
        """
        Return a copy of the manifold with the set tolerance value.
        This performs mesh simplification when the tolerance value is increased.
        """

    def as_original(self) -> Manifold:
        """
        This removes all relations (originalID, faceID, transform) to ancestor meshes
        and this new Manifold is marked an original. It also collapses colinear edges
        - these don't get collapsed at boundaries where originalID changes, so the
        reset may allow flat faces to be further simplified.
        """

    def is_empty(self) -> bool:
        """Does the Manifold have any triangles?"""

    def decompose(self) -> list[Manifold]:
        """
        This operation returns a vector of Manifolds that are topologically
        disconnected. If everything is connected, the vector is length one,
        containing a copy of the original. It is the inverse operation of Compose().
        """

    def split(self, cutter: Manifold) -> tuple[Manifold, Manifold]:
        """
        Split cuts this manifold in two using the cutter manifold. The first result
        is the intersection, second is the difference. This is more efficient than
        doing them separately.
        :param cutter:
        """

    def split_by_plane(self, normal: Doublex3, origin_offset: float) -> tuple[Manifold, Manifold]:
        """
        Convenient version of Split() for a half-space.
        :param normal: This vector is normal to the cutting plane and its length does
        not matter. The first result is in the direction of this vector, the second
        result is on the opposite side.
        :param origin_offset: The distance of the plane from the origin in the
        direction of the normal vector.
        """

    def trim_by_plane(self, normal: Doublex3, origin_offset: float) -> Manifold:
        """
        Identical to SplitByPlane(), but calculating and returning only the first
        result.
        :param normal: This vector is normal to the cutting plane and its length does
        not matter. The result is in the direction of this vector from the plane.
        :param origin_offset: The distance of the plane from the origin in the
        direction of the normal vector.
        """

    def slice(self, height: float) -> CrossSection:
        """
        Returns the cross section of this object parallel to the X-Y plane at the
        specified Z height, defaulting to zero. Using a height equal to the bottom of
        the bounding box will return the bottom faces, while using a height equal to
        the top of the bounding box will return empty.
        """

    def project(self) -> CrossSection:
        """
        Returns polygons representing the projected outline of this object
        onto the X-Y plane. These polygons will often self-intersect, so it is
        recommended to run them through the positive fill rule of CrossSection to get
        a sensible result before using them.
        """

    def status(self) -> Error:
        """
        Returns the reason for an input Mesh producing an empty Manifold. This Status
        only applies to Manifolds newly-created from an input Mesh - once they are
        combined into a new Manifold via operations, the status reverts to NoError,
        simply processing the problem mesh as empty. Likewise, empty meshes may still
        show NoError, for instance if they are small enough relative to their
        tolerance to be collapsed to nothing.
        """

    def bounding_box(self) -> tuple:
        """
        Gets the manifold bounding box as a tuple (xmin, ymin, zmin, xmax, ymax, zmax).
        """

    @overload
    @staticmethod
    def smooth(mesh: Mesh, sharpened_edges: Sequence[int] = [], edge_smoothness: Sequence[float] = []) -> Manifold:
        """
        Constructs a smooth version of the input mesh by creating tangents; this
        method will throw if you have supplied tangents with your mesh already. The
        actual triangle resolution is unchanged; use the Refine() method to
        interpolate to a higher-resolution curve.
        By default, every edge is calculated for maximum smoothness (very much
        approximately), attempting to minimize the maximum mean Curvature magnitude.
        No higher-order derivatives are considered, as the interpolation is
        independent per triangle, only sharing constraints on their boundaries.
        :param mesh: input Mesh.
        :param sharpened_edges: If desired, you can supply a vector of sharpened
        halfedges, which should in general be a small subset of all halfedges. Order
        of entries doesn't matter, as each one specifies the desired smoothness
        (between zero and one, with one the default for all unspecified halfedges)
        and the halfedge index (3 * triangle index + [0,1,2] where 0 is the edge
        between triVert 0 and 1, etc).
        :param edge_smoothness: Smoothness values associated to each halfedge defined
        in sharpened_edges. At a smoothness value of zero, a sharp crease is made. The
        smoothness is interpolated along each edge, so the specified value should be
        thought of as an average. Where exactly two sharpened edges meet at a vertex,
        their tangents are rotated to be colinear so that the sharpened edge can be
        continuous. Vertices with only one sharpened edge are completely smooth,
        allowing sharpened edges to smoothly vanish at termination. A single vertex
        can be sharpened by sharping all edges that are incident on it, allowing cones
        to be formed.
        """

    @overload
    @staticmethod
    def smooth(mesh: Mesh64, sharpened_edges: Sequence[int] = [], edge_smoothness: Sequence[float] = []) -> Manifold: ...

    @staticmethod
    def batch_boolean(manifolds: Sequence[Manifold], op: OpType) -> Manifold:
        """
        Perform the given boolean operation on a list of Manifolds. In case of
        Subtract, all Manifolds in the tail are differenced from the head.
        """

    @staticmethod
    def compose(manifolds: Sequence[Manifold]) -> Manifold:
        """
        Constructs a new manifold from a vector of other manifolds. This is a purely
        topological operation, so care should be taken to avoid creating
        overlapping results. It is the inverse operation of Decompose().
        :param manifolds: A vector of Manifolds to lazy-union together.
        """

    @staticmethod
    def tetrahedron() -> Manifold:
        """
        Constructs a tetrahedron centered at the origin with one vertex at (1,1,1)
        and the rest at similarly symmetric points.
        """

    @staticmethod
    def cube(size: Doublex3 = (1.0, 1.0, 1.0), center: bool = False) -> Manifold:
        """
        Constructs a unit cube (edge lengths all one), by default in the first
        octant, touching the origin. If any dimensions in size are negative, or if
        all are zero, an empty Manifold will be returned.
        :param size: The X, Y, and Z dimensions of the box.
        :param center: Set to true to shift the center to the origin.
        """

    @staticmethod
    def extrude(crossSection: CrossSection, height: float, n_divisions: int = 0, twist_degrees: float = 0.0, scale_top: Doublex2 = (1.0, 1.0)) -> Manifold:
        """
        Constructs a manifold from a set of polygons by extruding them along the
        Z-axis.
        Note that high twistDegrees with small nDivisions may cause
        self-intersection. This is not checked here and it is up to the user to
        choose the correct parameters.
        :param cross_section: A set of non-overlapping polygons to extrude.
        :param height: Z-extent of extrusion.
        :param n_divisions: Number of extra copies of the crossSection to insert into
        the shape vertically; especially useful in combination with twistDegrees to
        avoid interpolation artifacts. Default is none.
        :param twist_degrees: Amount to twist the top crossSection relative to the
        bottom, interpolated linearly for the divisions in between.
        :param scale_top: Amount to scale the top (independently in X and Y). If the
        scale is {0, 0}, a pure cone is formed with only a single vertex at the top.
        Note that scale is applied after twist.
        Default {1, 1}.
        """

    @staticmethod
    def revolve(crossSection: CrossSection, circular_segments: int = 0, revolve_degrees: float = 360.0) -> Manifold:
        """
        Constructs a manifold from a set of polygons by revolving this cross-section
        around its Y-axis and then setting this as the Z-axis of the resulting
        manifold. If the polygons cross the Y-axis, only the part on the positive X
        side is used. Geometrically valid input will result in geometrically valid
        output.
        :param cross_section: A set of non-overlapping polygons to revolve.
        :param circular_segments: Number of segments along its diameter. Default is
        calculated by the static Defaults.
        :param revolve_degrees: Number of degrees to revolve. Default is 360 degrees.
        """

    @staticmethod
    def level_set(f: Callable[[float, float, float], float], bounds: Sequence[float], edgeLength: float, level: float = 0.0, tolerance: float = -1) -> Manifold:
        """
        Constructs a level-set manifold from the input Signed-Distance Function
        (SDF). This uses a form of Marching Tetrahedra (akin to Marching
        Cubes, but better for manifoldness). Instead of using a cubic grid, it uses a
        body-centered cubic grid (two shifted cubic grids). These grid points are
        snapped to the surface where possible to keep short edges from forming.
        :param sdf: The signed-distance functor, containing this function signature:
        `double operator()(vec3 point)`, which returns the
        signed distance of a given point in R^3. Positive values are inside,
        negative outside. There is no requirement that the function be a true
        distance, or even continuous.
        :param bounds: An axis-aligned box that defines the extent of the grid.
        :param edge_length: Approximate maximum edge length of the triangles in the
        final result. This affects grid spacing, and hence has a strong effect on
        performance.
        :param level: Extract the surface at this value of your sdf; defaults to
        zero. You can inset your mesh by using a positive value, or outset it with a
        negative value.
        :param tolerance: Ensure each vertex is within this distance of the true
        surface. Defaults to -1, which will return the interpolated
        crossing-point based on the two nearest grid points. Small positive values
        will require more sdf evaluations per output vertex.
        :param can_parallel: Parallel policies violate will crash language runtimes
        with runtime locks that expect to not be called back by unregistered threads.
        This allows bindings use LevelSet despite being compiled with MANIFOLD_PAR
        active.
        """

    @staticmethod
    def cylinder(height: float, radius_low: float, radius_high: float = -1.0, circular_segments: int = 0, center: bool = False) -> Manifold:
        """
        A convenience constructor for the common case of extruding a circle. Can also
        form cones if both radii are specified.
        :param height: Z-extent
        :param radius_low: Radius of bottom circle. Must be positive.
        :param radius_high: Radius of top circle. Can equal zero. Default is equal to
        radiusLow.
        :param circular_segments: How many line segments to use around the circle.
        Default is calculated by the static Defaults.
        :param center: Set to true to shift the center to the origin. Default is
        origin at the bottom.
        """

    @staticmethod
    def sphere(radius: float, circular_segments: int = 0) -> Manifold:
        """
        Constructs a geodesic sphere of a given radius.
        :param radius: Radius of the sphere. Must be positive.
        :param circular_segments: Number of segments along its
        diameter. This number will always be rounded up to the nearest factor of
        four, as this sphere is constructed by refining an octahedron. This means
        there are a circle of vertices on all three of the axis planes. Default is
        calculated by the static Defaults.
        """

    @staticmethod
    def reserve_ids(n: int) -> int:
        """
        Returns the first of n sequential new unique mesh IDs for marking sets of
        triangles that can be looked up after further operations. Assign to
        Mesh.runOriginalID vector.
        """

class Mesh:
    def __init__(self, vert_properties: Annotated[ArrayLike, dict(dtype='float32', shape=(None, None), order='C')], tri_verts: Annotated[ArrayLike, dict(dtype='uint32', shape=(None, 3), order='C')], merge_from_vert: Annotated[ArrayLike, dict(dtype='uint32', shape=(None), order='C')] | None = None, merge_to_vert: Annotated[ArrayLike, dict(dtype='uint32', shape=(None), order='C')] | None = None, run_index: Annotated[ArrayLike, dict(dtype='uint32', shape=(None), order='C')] | None = None, run_original_id: Annotated[ArrayLike, dict(dtype='uint32', shape=(None), order='C')] | None = None, run_transform: Annotated[ArrayLike, dict(dtype='float32', shape=(None, 4, 3), order='C')] | None = None, face_id: Annotated[ArrayLike, dict(dtype='uint32', shape=(None), order='C')] | None = None, halfedge_tangent: Annotated[ArrayLike, dict(dtype='float32', shape=(None, 3, 4), order='C')] | None = None, tolerance: float = 0) -> None: ...

    @property
    def vert_properties(self) -> Annotated[ArrayLike, dict(dtype='float32', order='C', writable=False)]: ...

    @property
    def tri_verts(self) -> Annotated[ArrayLike, dict(dtype='int32', order='C', writable=False)]: ...

    @property
    def run_transform(self) -> Annotated[ArrayLike, dict(dtype='float32', order='C', writable=False)]: ...

    @property
    def halfedge_tangent(self) -> Annotated[ArrayLike, dict(dtype='float32', order='C', writable=False)]: ...

    @property
    def merge_from_vert(self) -> list[int]: ...

    @property
    def merge_to_vert(self) -> list[int]: ...

    @property
    def run_index(self) -> list[int]: ...

    @property
    def run_original_id(self) -> list[int]: ...

    @property
    def face_id(self) -> list[int]: ...

    def merge(self) -> bool:
        """
        Updates the mergeFromVert and mergeToVert vectors in order to create a
        manifold solid. If the Mesh is already manifold, no change will occur and
        the function will return false. Otherwise, this will merge verts along open
        edges within tolerance (the maximum of the Mesh tolerance and the
        baseline bounding-box tolerance), keeping any from the existing merge
        vectors, and return true.
        There is no guarantee the result will be manifold - this is a best-effort
        helper function designed primarily to aid in the case where a manifold
        multi-material Mesh was produced, but its merge vectors were lost due to
        a round-trip through a file format. Constructing a Manifold from the result
        will report an error status if it is not manifold.
        """

class Mesh64:
    def __init__(self, vert_properties: Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='C')], tri_verts: Annotated[ArrayLike, dict(dtype='uint64', shape=(None, 3), order='C')], merge_from_vert: Annotated[ArrayLike, dict(dtype='uint64', shape=(None), order='C')] | None = None, merge_to_vert: Annotated[ArrayLike, dict(dtype='uint64', shape=(None), order='C')] | None = None, run_index: Annotated[ArrayLike, dict(dtype='uint64', shape=(None), order='C')] | None = None, run_original_id: Annotated[ArrayLike, dict(dtype='uint32', shape=(None), order='C')] | None = None, run_transform: Annotated[ArrayLike, dict(dtype='float64', shape=(None, 4, 3), order='C')] | None = None, face_id: Annotated[ArrayLike, dict(dtype='uint64', shape=(None), order='C')] | None = None, halfedge_tangent: Annotated[ArrayLike, dict(dtype='float64', shape=(None, 3, 4), order='C')] | None = None, tolerance: float = 0) -> None: ...

    @property
    def vert_properties(self) -> Annotated[ArrayLike, dict(dtype='float64', order='C', writable=False)]: ...

    @property
    def tri_verts(self) -> Annotated[ArrayLike, dict(dtype='uint64', order='C', writable=False)]: ...

    @property
    def run_transform(self) -> Annotated[ArrayLike, dict(dtype='float64', order='C', writable=False)]: ...

    @property
    def halfedge_tangent(self) -> Annotated[ArrayLike, dict(dtype='float64', order='C', writable=False)]: ...

    @property
    def merge_from_vert(self) -> list[int]: ...

    @property
    def merge_to_vert(self) -> list[int]: ...

    @property
    def run_index(self) -> list[int]: ...

    @property
    def run_original_id(self) -> list[int]: ...

    @property
    def face_id(self) -> list[int]: ...

    def merge(self) -> bool:
        """
        Updates the mergeFromVert and mergeToVert vectors in order to create a
        manifold solid. If the Mesh is already manifold, no change will occur and
        the function will return false. Otherwise, this will merge verts along open
        edges within tolerance (the maximum of the Mesh tolerance and the
        baseline bounding-box tolerance), keeping any from the existing merge
        vectors, and return true.
        There is no guarantee the result will be manifold - this is a best-effort
        helper function designed primarily to aid in the case where a manifold
        multi-material Mesh was produced, but its merge vectors were lost due to
        a round-trip through a file format. Constructing a Manifold from the result
        will report an error status if it is not manifold.
        """

class OpType(enum.Enum):
    """Operation types for batch_boolean"""

    Add = 0

    Subtract = 1

    Intersect = 2

def get_circular_segments(radius: float) -> int:
    """
    Determine the result of the SetMinCircularAngle(),
    SetMinCircularEdgeLength(), and SetCircularSegments() defaults.
    :param radius: For a given radius of circle, determine how many default
    segments there will be.
    """

def set_circular_segments(number: int) -> None:
    """
    Sets the default number of circular segments for the
    CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and
    Manifold::Revolve() constructors. Overrides the edge length and angle
    constraints and sets the number of segments to exactly this value.
    :param number: Number of circular segments. Default is 0, meaning no
    constraint is applied.
    """

def set_min_circular_angle(angle: float) -> None:
    """
    Sets an angle constraint the default number of circular segments for the
    CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and
    Manifold::Revolve() constructors. The number of segments will be rounded up
    to the nearest factor of four.
    :param angle: The minimum angle in degrees between consecutive segments. The
    angle will increase if the the segments hit the minimum edge length.
    Default is 10 degrees.
    """

def set_min_circular_edge_length(length: float) -> None:
    """
    Sets a length constraint the default number of circular segments for the
    CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and
    Manifold::Revolve() constructors. The number of segments will be rounded up
    to the nearest factor of four.
    :param length: The minimum length of segments. The length will
    increase if the the segments hit the minimum angle. Default is 1.0.
    """

def triangulate(polygons: Sequence[DoubleNx2], epsilon: float = -1) -> IntNx3:
    """
    @brief Triangulates a set of &epsilon;-valid polygons. If the input is not
    &epsilon;-valid, the triangulation may overlap, but will always return a
    manifold result that matches the input edge directions.
    :param polygons: The set of polygons, wound CCW and representing multiple
    polygons and/or holes.
    :param epsilon: The value of &epsilon;, bounding the uncertainty of the
    input.
    @return std::vector<ivec3> The triangles, referencing the original
    polygon points in order.
    """
