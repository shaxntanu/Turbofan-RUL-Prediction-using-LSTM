import { useState } from "react";
import "./engine-anatomy.css";

const SECTIONS = [
  { id: "fan", label: "Fan", detail: "The front fan sends most inlet air around the core and the rest into it." },
  { id: "compressor", label: "Compressor", detail: "Rows of compressor blades raise the pressure of air entering the core." },
  { id: "combustor", label: "Combustor", detail: "Fuel burns with compressed core air here, adding energy to the flow." },
  { id: "turbine", label: "Turbine", detail: "Turbine stages extract energy from hot gas to drive the engine spools." },
  { id: "nozzle", label: "Nozzle", detail: "The remaining core flow expands through the exhaust nozzle." },
];

const FAN_BLADES = Array.from({ length: 14 }, (_, index) => index * (360 / 14));
const COMPRESSOR_BLADES = Array.from({ length: 8 }, (_, index) => index);
const TURBINE_BLADES = Array.from({ length: 7 }, (_, index) => index);
const FASTENERS = [
  [114, 211], [154, 225], [194, 211], [292, 249], [352, 249], [505, 249],
  [565, 249], [634, 238], [685, 230], [744, 227],
];

function SelectableSection({ id, active, onSelect, children, className = "" }) {
  const section = SECTIONS.find((item) => item.id === id);
  return (
    <g
      className={`engine-part ${active === id ? "engine-part-active" : ""} ${className}`}
      role="button"
      tabIndex="0"
      aria-label={section.label}
      aria-pressed={active === id}
      onMouseEnter={() => onSelect(id)}
      onFocus={() => onSelect(id)}
      onClick={() => onSelect(id)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect(id);
        }
      }}
    >
      {children}
    </g>
  );
}

export default function EngineAnatomy({ dataset, theme }) {
  const [view, setView] = useState("engine");
  const [active, setActive] = useState("fan");
  const selected = SECTIONS.find((section) => section.id === active);
  const calibrationSrc = `${import.meta.env.BASE_URL}conformal-calibration.html?${new URLSearchParams({ dataset, theme })}`;

  return (
    <section className="engine-anatomy" aria-labelledby="engine-anatomy-title">
      <div className="engine-anatomy-head">
        <div>
          <div className="eyebrow">{view === "engine" ? "ENGINE CUTAWAY" : "ANATOMY FIGURE"}</div>
          <h2 id="engine-anatomy-title">{view === "engine" ? "Turbofan anatomy" : "Split-conformal RUL intervals"}</h2>
        </div>
        <div className="engine-model-toggle" role="group" aria-label="Choose model view">
          <button type="button" className={view === "engine" ? "selected" : ""} aria-pressed={view === "engine"} onClick={() => setView("engine")}>Engine</button>
          <button type="button" className={view === "anatomy" ? "selected" : ""} aria-pressed={view === "anatomy"} onClick={() => setView("anatomy")}>Anatomy</button>
        </div>
      </div>

      {view === "engine" ? (
        <>
      <svg className="engine-drawing" viewBox="0 0 820 300" role="group" aria-label="Interactive schematic of a turbofan engine mounted on a test stand">
        <ellipse className="engine-shadow" cx="416" cy="260" rx="328" ry="16" />

        <g aria-hidden="true">
          <path className="engine-base-side" d="M83 239 690 239 755 222 147 222Z" />
          <path className="engine-base-top" d="M83 229 690 229 755 212 147 212Z" />
          <path className="engine-base-edge" d="M83 229v10h607v-10M690 239l65-17v-10l-65 17" />
          <path className="engine-rail" d="M117 219 681 219M144 211 706 211" />
          <path className="engine-support" d="m230 215 16-42 34 0 18 42zM532 215l17-47 39 0 20 47z" />
          <path className="engine-support-face" d="m230 215 16-42h12l-12 42zM532 215l17-47h12l-11 47z" />
          <path className="engine-mount" d="M222 208h82v9h-82zM524 208h92v9h-92z" />
          {FASTENERS.map(([x, y]) => <circle key={`${x}-${y}`} className="engine-fastener" cx={x} cy={y} r="2.2" />)}
        </g>

        <g aria-hidden="true">
          <path className="engine-bypass-top" d="M143 144Q173 70 250 70h425q34 0 53 35l-25 11q-15-22-36-22H252q-60 0-86 55Z" />
          <path className="engine-bypass-bottom" d="M146 191q28 54 88 54h395q58 0 84-39l-22-10q-22 29-65 29H236q-43 0-66-41Z" />
          <path className="engine-cowl-lip" d="M127 157q0-51 39-74l15 11q-29 24-29 63t29 61l-15 10q-39-22-39-71Z" />
          <path className="engine-nozzle-shell" d="M671 112q35 5 56 29v31q-21 21-56 27l-8-13q26-7 42-24v-13q-16-18-42-24Z" />
          <path className="engine-cowl-seam" d="M177 91q24-19 72-19M180 224q29 20 70 20M672 105q35 4 58 27M674 200q33-6 56-27" />
        </g>

        <SelectableSection id="fan" active={active} onSelect={setActive}>
          <ellipse className="engine-fan-ring" cx="170" cy="159" rx="37" ry="69" />
          <ellipse className="engine-fan-face" cx="170" cy="159" rx="30" ry="59" />
          <g className="engine-rotor">
            {FAN_BLADES.map((angle) => (
              <path key={angle} className="engine-fan-blade" transform={`rotate(${angle} 170 159)`} d="M168 155Q164 135 154 109Q167 119 173 151Z" />
            ))}
            <ellipse className="engine-fan-hub" cx="170" cy="159" rx="11" ry="18" />
            <path className="engine-fan-cap" d="M170 143q10 7 10 16t-10 16q-10-7-10-16t10-16Z" />
          </g>
          <ellipse className="engine-fan-outline" cx="170" cy="159" rx="36" ry="68" />
        </SelectableSection>

        <g aria-hidden="true">
          <path className="engine-core-shell" d="M207 118q30-22 74-22h355q26 0 44 19v88q-20 20-48 20H264q-35 0-57-24Z" />
          <path className="engine-core-edge" d="M210 121q27-20 70-20h355q23 0 40 16M208 198q25 20 57 20h369q27 0 45-18" />
          <path className="engine-shaft" d="M205 158h505v9H205z" />
          <path className="engine-shaft-light" d="M211 159h492" />
          <path className="engine-bypass-divider" d="M218 119q22-12 47-12v100q-28 0-49-12Z" />
          <path className="engine-divider-rib" d="M256 111v93m4-93v93" />
        </g>

        <SelectableSection id="compressor" active={active} onSelect={setActive}>
          {[286, 326, 366].map((x, stage) => (
            <g key={x} className="engine-stage">
              <path className="engine-stage-disc" d={`M${x - 7} 126q18-11 28 0v67q-12 11-28 0Z`} />
              {COMPRESSOR_BLADES.map((row) => {
                const y = 130 + row * 7.1;
                return <path key={`${x}-${row}`} className="engine-compressor-blade" d={`M${x + 1} ${y}l${stage % 2 ? 7 : 9} 2-4 5-5-2Z`} />;
              })}
              <path className="engine-stage-rim" d={`M${x + 23} 126v67`} />
            </g>
          ))}
          {[313, 353].flatMap((x) => COMPRESSOR_BLADES.map((row) => {
            const y = 131 + row * 7.1;
            return <path key={`${x}-${row}`} className="engine-stator-blade" d={`M${x} ${y}l4 3-2 4-3-3Z`} />;
          }))}
        </SelectableSection>

        <SelectableSection id="combustor" active={active} onSelect={setActive}>
          <path className="engine-combustor-case" d="M414 119q8-12 22-12h82q16 0 23 14v61q-7 15-23 15h-82q-15 0-22-12Z" />
          <path className="engine-combustor-liner" d="M425 128q6-8 14-8h72q9 0 15 10v51q-6 9-15 9h-72q-9 0-14-8Z" />
          <path className="engine-combustor-rib" d="M438 122v65m13-67v69m51-69v69m12-66v63" />
          {[0, 1, 2, 3, 4].map((index) => <circle key={index} className="engine-injector" cx={441 + index * 15} cy="112" r="2.4" />)}
          <path className="engine-combustor-top" d="M430 111h91l8 6h-105Z" />
        </SelectableSection>

        <SelectableSection id="turbine" active={active} onSelect={setActive}>
          {[566, 605].map((x, stage) => (
            <g key={x} className="engine-stage">
              <path className="engine-turbine-disc" d={`M${x - 6} 125q15-10 25 0v68q-10 11-25 0Z`} />
              {TURBINE_BLADES.map((row) => {
                const y = 132 + row * 9;
                return <path key={`${x}-${row}`} className="engine-turbine-blade" d={`M${x} ${y}l${stage ? 9 : 11} 3-8 7-5-3Z`} />;
              })}
              <path className="engine-stage-rim" d={`M${x + 22} 126v67`} />
            </g>
          ))}
          <path className="engine-nozzle-core" d="M632 137q31 8 52 21-21 16-52 23l5-13q17-5 27-10-10-7-27-11Z" />
          <path className="engine-nozzle-line" d="M637 142q20 6 34 16-14 10-34 16" />
        </SelectableSection>

        <SelectableSection id="nozzle" active={active} onSelect={setActive} className="engine-nozzle-focus">
          <path className="engine-nozzle-lip" d="M704 140q19 9 30 19-11 12-30 19l-6-9q12-5 20-10-8-7-20-11Z" />
          <path className="engine-nozzle-edge" d="M730 153q10 6 0 13" />
        </SelectableSection>

        <g aria-hidden="true" className="engine-detail-lines">
          <path d="M246 115v-17m8 18v-16m18 12v-17m175 93h14m15 0h14m15 0h14m15 0h14" />
          <path d="M267 224v-12m6 12v-12m9 13v-12m334 8v-10m9 10v-10m9 9v-10" />
          <circle cx="395" cy="119" r="2.6" /><circle cx="397" cy="194" r="2.6" />
          <circle cx="548" cy="119" r="2.6" /><circle cx="548" cy="194" r="2.6" />
        </g>
      </svg>

      <div className="engine-anatomy-foot">
        <p className="engine-readout" aria-live="polite"><b>{selected.label}</b><span>{selected.detail}</span></p>
        <div className="engine-part-select" role="group" aria-label="Select an engine section">
          {SECTIONS.map((section) => (
            <button
              key={section.id}
              type="button"
              className={active === section.id ? "engine-part-button selected" : "engine-part-button"}
              aria-pressed={active === section.id}
              onClick={() => setActive(section.id)}
            >
              {section.label}
            </button>
          ))}
        </div>
      </div>
      <p className="engine-anatomy-note">Generic schematic, not to scale. C-MAPSS sensor channels are not assigned to physical engine parts.</p>
        </>
      ) : (
        <iframe
          className="engine-anatomy-frame"
          src={calibrationSrc}
          title={`Anatomy calibration model for ${dataset}`}
          loading="lazy"
          referrerPolicy="no-referrer"
        />
      )}
    </section>
  );
}
