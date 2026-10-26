// Inference for the exported Gradient Boosting model (MODEL comes from model_export.js, written by export_model.py).
// Shared by the page and by tests/test_browser_model.py, which checks it against scikit-learn.
// scikit-learn's trees compare features as 32-bit floats, so round each value the same way (Math.fround).
// Without this, a value that sits between a float32 and its float64 neighbour can take the other branch.
function vector(s){
  const v = MODEL.num.map(k => Math.fround(k==="hop_distance" ? Math.abs(s.source_zone-s.protected_zone) : +s[k]));
  MODEL.cats.forEach(c => v.push(s.actuator_type===c ? 1 : 0));
  return v;
}
function predictProba(s){
  const x = vector(s), r = MODEL.init.slice();
  for (const stage of MODEL.trees){
    for (let k=0;k<stage.length;k++){
      const [f,th,l,rr,val] = stage[k]; let n=0;
      while (l[n] !== -1) n = x[f[n]] <= th[n] ? l[n] : rr[n];
      r[k] += MODEL.lr * val[n];
    }
  }
  const m = Math.max(...r), e = r.map(z=>Math.exp(z-m)), t = e.reduce((a,b)=>a+b,0);
  const out = {}; MODEL.classes.forEach((c,i)=>out[c]=e[i]/t); return out;
}
function gated(p){
  if (p.Protective >= MODEL.tau) return "Protective";
  let best=null; for (const c of MODEL.classes) if (c!=="Protective" && (best===null || p[c]>p[best])) best=c;
  return best;
}
function hardConstraint(s){
  if (s.occupancy && s.airflow_direction_to_protected)
    return "Airflow is moving toward an occupied protected zone. Any push would carry the hazard onto people, so the action is rejected whatever the model predicts.";
  return null;
}
if (typeof module !== "undefined") module.exports = { vector, predictProba, gated, hardConstraint };
