import { useState } from "react";

const OPTIONS = [
  "Rarely",
  "1–2 times/week",
  "3–4 times/week",
  "5+ times/week",
];

export function FrequencySlider({ value, onChange }) {
  const index = OPTIONS.indexOf(value);
  const selected = index >= 0 ? index : 0;

  const handleChange = (e) => {
    const i = Number(e.target.value);
    onChange(OPTIONS[i]);
  };

  return (
    <div className="freq-slider">
      <div className="freq-labels">
        {OPTIONS.map((label, i) => (
          <label
            key={label}
            className={i === selected ? "freq-label active" : "freq-label"}
          >
            <input
              type="radio"
              name="frequency"
              value={i}
              checked={i === selected}
              onChange={handleChange}
            />
            {label}
          </label>
        ))}
      </div>
      <div className="freq-track">
        <div
          className="freq-thumb"
          style={{ left: `${(selected / (OPTIONS.length - 1)) * 100}%` }}
        />
      </div>
      <div className="freq-value">{OPTIONS[selected]}</div>
    </div>
  );
}
