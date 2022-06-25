module IrelandCovidSim
    function project_path(parts...) 
        return normpath(@__DIR__, "..", parts...)
    end
    export project_path 
end
