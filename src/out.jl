# ============================================================
# Output Methods
# ============================================================

function KitBase.plot_line(KS::AbstractSolverSet, uq::AbstractUQ, sol::AbstractSolution1D)

    x = deepcopy(KS.pSpace.x[1:KS.pSpace.nx])
    flowMean = zeros(KS.pSpace.nx, size(sol.w[1], 1))
    flowVar = similar(flowMean)

    for i = 1:KS.pSpace.nx
        flowMean[i, 1, 1] = mean(ctr[i].prim[1, :, 1], uq.op)
        flowMean[i, 1, 2] = mean(ctr[i].prim[1, :, 2], uq.op)
        flowVar[i, 1, 1] = var(ctr[i].prim[1, :, 1], uq.op)
        flowVar[i, 1, 2] = var(ctr[i].prim[1, :, 2], uq.op)
        for j = 2:4
            for k = 1:2
                flowMean[i, j, k] = mean(ctr[i].prim[j, :, k], uq.op)
                flowVar[i, j, k] = var(ctr[i].prim[j, :, k], uq.op)
            end
        end
        flowMean[i, 5, 1] = mean(lambda_tchaos(ctr[i].prim[5, :, 1], KS.gas.mi, uq), uq.op)
        flowVar[i, 5, 1] = var(lambda_tchaos(ctr[i].prim[5, :, 1], KS.gas.mi, uq), uq.op)
    end

    xlabel("X")
    ylabel("Expectation")
    legend("Ni", "Ne", "Ui", "Ue", "Ti", "Te")
    p1 = plot(x, flowMean[:, 1, 1])
    p1 = oplot(x, flowMean[:, 1, 2])
    p1 = oplot(x, flowMean[:, 2, 1])
    p1 = oplot(x, flowMean[:, 2, 2])
    p1 = oplot(x, flowMean[:, 5, 1])
    p1 = oplot(x, flowMean[:, 5, 2])
    display(p1)

    xlabel("X")
    ylabel("Expectation")
    legend("Ex", "Ey", "Ez", "Bx", "By", "Bz")
    p2 = plot(x, emMean[:, 1])
    p2 = oplot(x, emMean[:, 2])
    p2 = oplot(x, emMean[:, 3])
    p2 = oplot(x, emMean[:, 4])
    p2 = oplot(x, emMean[:, 5])
    p2 = oplot(x, emMean[:, 6])
    display(p2)

    xlabel("X")
    ylabel("Variance")
    legend("Ni", "Ne", "Ui", "Ue", "Ti", "Te")
    p3 = plot(x, flowVar[:, 1, 1])
    p3 = oplot(x, flowVar[:, 1, 2])
    p3 = oplot(x, flowVar[:, 2, 1])
    p3 = oplot(x, flowVar[:, 2, 2])
    p3 = oplot(x, flowVar[:, 5, 1])
    p3 = oplot(x, flowVar[:, 5, 2])
    display(p3)

    xlabel("X")
    ylabel("Variance")
    legend("Ex", "Ey", "Ez", "Bx", "By", "Bz")
    p4 = plot(x, emVar[:, 1])
    p4 = oplot(x, emVar[:, 2])
    p4 = oplot(x, emVar[:, 3])
    p4 = oplot(x, emVar[:, 4])
    p4 = oplot(x, emVar[:, 5])
    p4 = oplot(x, emVar[:, 6])
    display(p4)

end


function KitBase.write_jld(
    KS::AbstractSolverSet,
    ctr::AbstractArray{<:AbstractControlVolume,1},
    uq::AbstractUQ,
    t = 0::Real,
)

    strIter = string(t)
    fileOut = KS.outputFolder * "data/t=" * strIter * ".jld2"

    @save fileOut KS ctr uq t

end
