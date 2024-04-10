import plotly.graph_objects as go
import torch


# 线阵，等间距，等幅同相
# dna(1,1,L) Fdb(delta,)
def pattern(dna: torch.Tensor, l: float, d: float, delta: int, theta_0: float) -> None:
    M = dna.shape[2]
    ex = dna.reshape(M,)
    k = 2*torch.pi/l
    theta_0 = (torch.tensor(theta_0)*torch.pi)/180

    theta = torch.linspace(-torch.pi/2, torch.pi/2, delta)
    F = torch.zeros((delta,))
    for i in range(delta):
        phi = (k*d * torch.arange(
            0, M))*(torch.sin(theta[i])-torch.sin(theta_0)).to(dtype=torch.float)
        temp = torch.exp(torch.complex(
            torch.zeros(M,).to(dtype=torch.float), phi))
        F[i] = torch.abs(torch.sum(temp*ex))
    Fdb = 20*torch.log10(F/F.max())
    return Fdb


def plot(Fdb: torch.Tensor, delta: int, theta_min: float, theta_max: float):
    theta = torch.linspace(theta_min, theta_max, delta)
    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=theta,
            y=Fdb,
        )
    )
    fig.update_layout(
        template="simple_white",
        title="方向图",
        xaxis_title="theta",
        xaxis_range=[theta_min-10, theta_max+10],
        yaxis_title="Fdb",
        yaxis_range=[-60, 0.5],
    )
    fig.show()


if __name__ == "__main__":
    pass
