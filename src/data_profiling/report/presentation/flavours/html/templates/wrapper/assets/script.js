/* Native chart tooltips and line-chart crosshair.
 * Charts are inline SVG with a <title> per mark (works without JavaScript).
 * This upgrades them to rich tooltips; lines are separated by " · " and a
 * first line without ": " is the heading. Line charts carry their (already
 * downsampled) data in data-dp-line for the crosshair. */
(() => {
    const SEP = " · ";
    const NS = "http://www.w3.org/2000/svg";
    let tip = null;

    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
    const row = (k, v, color) =>
        `<div class="dp-tt-row"><span class="dp-tt-k">${color ? `<span class="dp-tt-sw" style="background:${esc(color)}"></span>` : ""}${esc(k)}</span>` +
        `<span class="dp-tt-v">${esc(v)}</span></div>`;
    const textToHtml = (text) => {
        const lines = text.split(SEP);
        let html = "";
        if (lines.length && lines[0].indexOf(": ") < 0) {
            html += `<div class="dp-tt-head">${esc(lines.shift())}</div>`;
        }
        return html + lines.map((line) => {
            const i = line.indexOf(": ");
            return i < 0 ? `<div class="dp-tt-row">${esc(line)}</div>` : row(line.slice(0, i), line.slice(i + 2));
        }).join("");
    };

    const show = (html, ev) => {
        tip.innerHTML = html;
        tip.hidden = false;
        const pad = 14;
        const r = tip.getBoundingClientRect();
        let x = ev.clientX + pad;
        let y = ev.clientY + pad;
        if (x + r.width > window.innerWidth - 4) x = ev.clientX - r.width - pad;
        if (y + r.height > window.innerHeight - 4) y = ev.clientY - r.height - pad;
        tip.style.transform = `translate(${Math.max(4, x)}px, ${Math.max(4, y)}px)`;
    };
    const hide = () => { if (tip) tip.hidden = true; };

    const numFmt = new Intl.NumberFormat(undefined, {maximumSignificantDigits: 5});
    const bigFmt = new Intl.NumberFormat(undefined, {notation: "compact", maximumSignificantDigits: 4});
    const fmtNum = (v) => {
        if (v === null || v === undefined || Number.isNaN(v)) return "–";
        const a = Math.abs(v);
        if (a >= 1e6) return bigFmt.format(v);
        if (a !== 0 && a < 1e-3) return v.toExponential(2);
        return numFmt.format(v);
    };
    const fmtDate = (sec, span) => {
        const iso = new Date(sec * 1000).toISOString();
        return span < 3 * 86400 ? iso.slice(0, 16).replace("T", " ") : iso.slice(0, 10);
    };
    const nearest = (xs, x) => {
        let lo = 0;
        let hi = xs.length - 1;
        while (hi - lo > 1) {
            const m = (lo + hi) >> 1;
            if (xs[m] < x) lo = m; else hi = m;
        }
        return Math.abs(xs[lo] - x) <= Math.abs(xs[hi] - x) ? lo : hi;
    };
    const toSvg = (svg, ev) => {
        const m = svg.getScreenCTM();
        return m ? new DOMPoint(ev.clientX, ev.clientY).matrixTransform(m.inverse()) : {x: -1, y: -1};
    };

    const setupLine = (svg, data) => {
        const [L, T, W, H] = data.p;
        const [x0, x1] = data.xd;
        const [y0, y1] = data.yd;
        const sx = (v) => L + ((v - x0) / ((x1 - x0) || 1)) * W;
        const sy = (v) => T + H - ((v - y0) / ((y1 - y0) || 1)) * H;
        const g = document.createElementNS(NS, "g");
        g.setAttribute("class", "dp-crosshair");
        g.setAttribute("pointer-events", "none");
        g.style.display = "none";
        const vline = document.createElementNS(NS, "line");
        vline.setAttribute("y1", T);
        vline.setAttribute("y2", T + H);
        vline.setAttribute("stroke", "currentColor");
        vline.setAttribute("stroke-opacity", "0.45");
        vline.setAttribute("stroke-dasharray", "3 3");
        g.appendChild(vline);
        const dots = data.s.map((s) => {
            const c = document.createElementNS(NS, "circle");
            c.setAttribute("r", "4");
            c.setAttribute("fill", s.c);
            c.setAttribute("stroke", "#fff");
            c.setAttribute("stroke-width", "2");
            g.appendChild(c);
            return c;
        });
        svg.appendChild(g);
        const leave = () => { g.style.display = "none"; hide(); };
        const move = (ev) => {
            const pt = toSvg(svg, ev);
            if (pt.x < L - 2 || pt.x > L + W + 2 || pt.y < T - 6 || pt.y > T + H + 6) return leave();
            const xv = x0 + ((pt.x - L) / W) * (x1 - x0);
            let hx = null;
            const rows = [];
            data.s.forEach((s, k) => {
                const xs = s.x || data.x;
                if (!xs || !xs.length) return;
                const i = nearest(xs, xv);
                const X = xs[i];
                const Y = s.y[i];
                if (hx === null) hx = X;
                if (Y === null) {
                    dots[k].style.display = "none";
                } else {
                    dots[k].style.display = "";
                    dots[k].setAttribute("cx", sx(X));
                    dots[k].setAttribute("cy", sy(Y));
                }
                rows.push(row(s.l || "Value", fmtNum(Y), data.s.length > 1 ? s.c : null));
            });
            if (hx === null) return leave();
            vline.setAttribute("x1", sx(hx));
            vline.setAttribute("x2", sx(hx));
            g.style.display = "";
            const head = data.d ? fmtDate(hx, x1 - x0) : fmtNum(hx);
            const gap = (data.g || []).find(([a, b]) => xv >= a && xv <= b);
            const gapRow = gap
                ? row("In gap", data.d ? `${fmtDate(gap[0], x1 - x0)} → ${fmtDate(gap[1], x1 - x0)}` : `${fmtNum(gap[0])} → ${fmtNum(gap[1])}`)
                : "";
            show(`<div class="dp-tt-head">${esc(head)}</div>${rows.join("")}${gapRow}`, ev);
        };
        svg.addEventListener("pointermove", move);
        svg.addEventListener("pointerdown", move);
        svg.addEventListener("pointerleave", leave);
    };

    const setupScatter = (svg, data) => {
        const [L, T, W, H] = data.p;
        const [x0, x1] = data.xd;
        const [y0, y1] = data.yd;
        const sx = (v) => L + ((v - x0) / ((x1 - x0) || 1)) * W;
        const sy = (v) => T + H - ((v - y0) / ((y1 - y0) || 1)) * H;
        const px = data.x.map(sx);
        const py = data.y.map(sy);
        const ring = document.createElementNS(NS, "circle");
        ring.setAttribute("r", "6");
        ring.setAttribute("fill", data.c);
        ring.setAttribute("stroke", "#fff");
        ring.setAttribute("stroke-width", "2");
        ring.setAttribute("pointer-events", "none");
        ring.style.display = "none";
        svg.appendChild(ring);
        const leave = () => { ring.style.display = "none"; hide(); };
        const move = (ev) => {
            const pt = toSvg(svg, ev);
            let best = -1;
            let bestD = 14 * 14; // hover radius, in chart units
            for (let i = 0; i < px.length; i++) {
                const dx = px[i] - pt.x;
                const dy = py[i] - pt.y;
                const d = dx * dx + dy * dy;
                if (d < bestD) { bestD = d; best = i; }
            }
            if (best < 0) return leave();
            ring.setAttribute("cx", px[best]);
            ring.setAttribute("cy", py[best]);
            ring.style.display = "";
            let same = 0;
            for (let i = 0; i < px.length; i++) if (data.x[i] === data.x[best] && data.y[i] === data.y[best]) same++;
            show(row(data.xl, fmtNum(data.x[best])) + row(data.yl, fmtNum(data.y[best])) +
                 (same > 1 ? row("Overlapping points", String(same)) : ""), ev);
        };
        svg.addEventListener("pointermove", move);
        svg.addEventListener("pointerdown", move);
        svg.addEventListener("pointerleave", leave);
    };

    const init = () => {
        const charts = document.querySelectorAll("svg.dp-chart");
        if (!charts.length) return;
        tip = document.createElement("div");
        tip.className = "dp-tooltip";
        tip.setAttribute("role", "tooltip");
        tip.hidden = true;
        document.body.appendChild(tip);
        charts.forEach((svg) => {
            // Move native titles into data attributes (no duplicate native tooltip).
            svg.querySelectorAll("title").forEach((t) => {
                if (t.parentNode !== svg) t.parentNode.setAttribute("data-dp-tip", t.textContent);
                t.remove();
            });
            const line = svg.getAttribute("data-dp-line");
            if (line) {
                try { setupLine(svg, JSON.parse(line)); return; } catch (e) { /* fall back to mark tooltips */ }
            }
            const scatter = svg.getAttribute("data-dp-scatter");
            if (scatter) {
                try { setupScatter(svg, JSON.parse(scatter)); return; } catch (e) { /* no point tooltips */ }
            }
            const move = (ev) => {
                const el = ev.target.closest("[data-dp-tip]");
                if (el && svg.contains(el)) show(textToHtml(el.getAttribute("data-dp-tip")), ev); else hide();
            };
            svg.addEventListener("pointermove", move);
            svg.addEventListener("pointerdown", move);
            svg.addEventListener("pointerleave", hide);
        });
        window.addEventListener("scroll", hide, {passive: true});
    };

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();

document.addEventListener("DOMContentLoaded", () => {
    /* Tooltip */
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => new bootstrap.Tooltip(el));

    /* Link */
    document.querySelectorAll("a:not([href^='#'])").forEach(el => el.setAttribute("target", "_blank"));

    /* Variables Dropdown */
    const variablesDropdown = document.querySelector("select#variables-dropdown");
    variablesDropdown.addEventListener("change", () => {
        const searchText = variablesDropdown.value.toLowerCase();

        document.querySelectorAll(".variable").forEach(el => {
            const title = el.firstElementChild.firstElementChild.firstElementChild.getAttribute("title").toLowerCase();
            let isMatch = title === (searchText);

            if (searchText === "") {
                isMatch = true
            }

            if (isMatch) {
                el.parentElement.classList.remove("d-none");
            } else {
                el.parentElement.classList.add("d-none");
            }
        });
    });

    /* Correlation */
    document.querySelector("#toggle-correlation-description")?.click(() => {
        document.querySelectorAll(".correlation-description").forEach(el => el.classList.toggle("d-none"));

        document.querySelectorAll(".tab-pane[id^=correlations_tab-]").forEach(el => {
            const diagramLength = el.querySelectorAll(".tab-pane[id^=correlations_tab-]").length;

            let classes = [];
            if (diagramLength === 1) {
                classes = ["col-sm-6", "col-sm-12"];
            } else if (diagramLength === 2) {
                classes = ["col-sm-4", "col-sm-6"];
            } else if (diagramLength === 3) {
                classes = ["col-sm-3", "col-sm-4"];
            } else {
                console.log("unsupported number of reports");
            }

            document.querySelectorAll(".correlation-diagram").forEach(el => {
                classes.forEach(clz => el.classList.toggle(clz));
            });
        });
    });
});
