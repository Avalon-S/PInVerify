# PIN-v2 Episode Filtration Gates Logic

```mermaid
graph TD
    %% Define Styles
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,stroke-dasharray: 5 5;
    classDef failure fill:#ffcdd2,stroke:#c62828,stroke-width:2px;
    classDef success fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px;
    classDef startend fill:#f5f5f5,stroke:#616161,stroke-width:2px,rx:10,ry:10;

    Input(Input Episode<br/>Start / Goal):::startend --> RunSim[<b>Simulation</b><br/>Run Agent with Shortest Path Oracle]:::process

    RunSim --> Gate1{<b>Gate 1: Cross-Floor?</b><br/><i>(Is Single Floor?)</i>}:::decision
    
    Gate1 -- "Yes (Pass)" --> Gate2{<b>Gate 2: Visible?</b><br/><i>(Mask Detected?)</i>}:::decision
    Gate1 -- "No (Fail)" --> FailG1[<b>Bad Episode</b><br/>Reason: Cross-Floor]:::failure

    subgraph "Gate 1 Details"
        direction TB
        G1_Method1[Method A: Path Height Range > 0.25m?]:::process
        G1_Method2[Method B: Trajectory Height Range > 0.25m?]:::process
        G1_Method1 -.-> |OR| G1_Combined((Result))
        G1_Method2 -.-> |OR| G1_Combined
    end
    
    Gate2 -- "Yes (Pass)" --> Gate3{<b>Gate 3: Height Diff?</b><br/><i>(Reachable Height?)</i>}:::decision
    Gate2 -- "No (Fail)" --> FailG2[<b>Bad Episode</b><br/>Reason: Target Invisible]:::failure
    
    subgraph "Gate 2 Details"
        direction TB
        G2_Check[Agent follows Shortest Path<br/>Check Semantic Mask in RGBD]:::process
    end

    Gate3 -- "Yes (Pass)" --> Success((<b>Good Episode</b><br/>Ready for Training)):::success
    Gate3 -- "No (Fail)" --> FailG3[<b>Bad Episode</b><br/>Reason: Height Out of Range<br/><i>(High > 1.6m or Low < 0m)</i>]:::failure

    subgraph "Gate 3 Details"
        direction TB
        G3_Calc[Calculate Diff:<br/>Goal_Y - Agent_Final_Y]:::process
    end

    %% Wiring Details to Main Flow (Conceptual)
    G1_Combined -.-> Gate1
    G2_Check -.-> Gate2
    G3_Calc -.-> Gate3
    
```
