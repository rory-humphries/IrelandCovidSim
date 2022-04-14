struct IGeometryPolygonSerialization
    data::String
end

struct IGeometryMultiPolygonSerialization
    data::String
end

JLD2.writeas(::Type{AG.IGeometry{AG.wkbPolygon}}) = IGeometryPolygonSerialization
JLD2.writeas(::Type{AG.IGeometry{AG.wkbMultiPolygon}}) = IGeometryMultiPolygonSerialization

function Base.convert(::Type{IGeometryPolygonSerialization}, g::AG.IGeometry{AG.wkbPolygon})
    return IGeometryPolygonSerialization(AG.toWKT(g))
end
function Base.convert(
    ::Type{IGeometryMultiPolygonSerialization}, g::AG.IGeometry{AG.wkbMultiPolygon}
)
    return IGeometryMultiPolygonSerialization(AG.toWKT(g))
end

function Base.convert(::Type{AG.IGeometry{AG.wkbPolygon}}, a::IGeometryPolygonSerialization)
    return AG.fromWKT(a.data)
end
function Base.convert(
    ::Type{AG.IGeometry{AG.wkbMultiPolygon}}, a::IGeometryMultiPolygonSerialization
)
    return AG.fromWKT(a.data)
end

