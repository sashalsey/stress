from firedrake import DirichletBC #type:ignore
import numpy as np
import warnings

def FixedDirichletBC(vectorFunctionSpaceCG, val, point):
    
    class FixOriginBC(DirichletBC):
        def __init__(self, V, val, subdomain, point):
            super().__init__(V, val, subdomain)
            self.point = point  # Store point for reconstruction
            
            sec = V.dm.getDefaultSection()
            dm = V.mesh().topology_dm
            coordsSection = dm.getCoordinateSection()
            coordsDM = dm.getCoordinateDM()
            dim = dm.getCoordinateDim()
            coordsVec = dm.getCoordinatesLocal()
            (vStart, vEnd) = dm.getDepthStratum(0)
            indices = []

            for pt in range(vStart, vEnd):
                x = dm.getVecClosure(coordsSection, coordsVec, pt).reshape(-1, dim).mean(axis=0)
                if np.allclose(point, x, atol=1e-6):  # More relaxed condition
                    print(f"Found matching node {pt} at {x} for point {point}")  # Debugging
                    if dm.getLabelValue("pyop2_ghost", pt) == -1:
                        indices.append(pt)

            nodes = []
            for i in indices:
                if sec.getDof(i) > 0:
                    nodes.append(sec.getOffset(i))

            self.nodes = np.asarray(nodes, dtype=int)
            if len(self.nodes) > 0:
                print(f"Fixing nodes: {self.nodes}")  # Debugging
            else:
                warnings.warn("Not fixing any nodes", UserWarning)

        # **Override the reconstruct method to properly handle adjoint operations**
        def reconstruct(self, V=None, g=None, sub_domain=None):
            V = V if V is not None else self.function_space()
            g = g if g is not None else self.g
            sub_domain = sub_domain if sub_domain is not None else self.sub_domain
            return FixOriginBC(V, g, sub_domain, self.point)  # Ensure `point` is passed

    return FixOriginBC(vectorFunctionSpaceCG, val, None, point)
