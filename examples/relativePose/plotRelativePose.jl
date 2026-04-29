function initRelativePosePlot()
    # Init plot
    set_theme!(theme_fra())
    fig = Figure(; size=(1100, 670));
    display(fig)
    axs = (
        GLMakie.Axis(fig[1, 1]; xlabel="Time [s]", ylabel="x [m]", limits=(0, nothing, nothing, nothing), title="Position estimation performance"),
        GLMakie.Axis(fig[2, 1]; xlabel="Time [s]", ylabel="y [m]", limits=(0, nothing, nothing, nothing)),
        GLMakie.Axis(fig[3, 1]; xlabel="Time [s]", ylabel="z [m]", limits=(0, nothing, nothing, nothing)),
        GLMakie.Axis(fig[1, 2]; xlabel="Time [s]", ylabel="vx [m/s]", limits=(0, nothing, nothing, nothing), title="Velocity estimation performance"),
        GLMakie.Axis(fig[2, 2]; xlabel="Time [s]", ylabel="vy [m/s]", limits=(0, nothing, nothing, nothing)),
        GLMakie.Axis(fig[3, 2]; xlabel="Time [s]", ylabel="vz [m/s]", limits=(0, nothing, nothing, nothing)),
        GLMakie.Axis(fig[1, 3]; xlabel="Time [s]", ylabel="θx [deg]", limits=(0, nothing, nothing, nothing), title="Attitude estimation performance"),
        GLMakie.Axis(fig[2, 3]; xlabel="Time [s]", ylabel="θy [deg]", limits=(0, nothing, nothing, nothing)),
        GLMakie.Axis(fig[3, 3]; xlabel="Time [s]", ylabel="θz [deg]", limits=(0, nothing, nothing, nothing)),        #GLMakie.Axis(fig[3, 1:3];
    )
    return axs
end

function plotRelativePose(axs, T, X, X̂, σ)
    T0 = zero(T)
    α = 0.5 + 0.5*rand()
    for i in 1:6
        plotNavError(axs[i], T, getindex.(X, i), getindex.(X̂, i), getindex.(σ, i), α)
    end

    qEst_IT = getindex.(X̂, [[7, 8, 9, 10]])
    qTrue_IT = getindex.(X, [[7, 8, 9, 10]])
    qErr_IB = q_attitudeError.(qTrue_IT, qEst_IT)
    for i in 7:9
        plotNavError(axs[i], T, getindex.(qErr_IB, i-6)*180/π, T0, getindex.(σ, i)*180/π, α)
    end
    return nothing
end

function plotNavError(ax, T, X, X̂, σ, α=1.0)
    lines!(ax, T, X - X̂; color=:white, alpha=α)
    lines!(ax, T, +3σ; linewidth=2, color=:red)
    lines!(ax, T, -3σ; linewidth=2, color=:red)
    return nothing
end
        #lines!(axs[7], getindex.(X, 1), getindex.(X, 3))
