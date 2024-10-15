document.getElementById("load-btn").onclick = async function () {
    const url = document.getElementById("url").value;
    const response = await fetch("/load", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ url: url })
    });

    const result = await response.json();
    document.getElementById("result").innerText = result.message || "Data loaded successfully.";
};

document.getElementById("query-btn").onclick = async function () {
    const query = document.getElementById("query").value;
    const response = await fetch("/query", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: query })
    });

    const result = await response.json();
    document.getElementById("result").innerText = result.answer || "No answer found.";
};

document.getElementById("clear-btn").onclick = async function () {
    // Clear stored data and output
    await fetch("/clear", { method: "POST" });
    document.getElementById("url").value = "";
    document.getElementById("query").value = "";
    document.getElementById("result").innerText = "Data cleared successfully.";
};
