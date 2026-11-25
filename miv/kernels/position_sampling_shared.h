#ifndef OMLX_POSITION_SAMPLING_SHARED_H
#define OMLX_POSITION_SAMPLING_SHARED_H
inline ulong position_mix(ulong v) {
  v=(v^(v>>30))*0xBF58476D1CE4E5B9ul;
  v=(v^(v>>27))*0x94D049BB133111EBul;return v^(v>>31);
}
inline float position_uniform(ulong seed,ulong position,uint token,ulong purpose) {
  ulong bits=position_mix(seed^purpose);
  bits=position_mix(bits^(position*0xD1B54A32D192ED03ul));
  bits=position_mix(bits^ulong(token));
  return (float(bits>>41)+0.5f)*0x1p-23f;
}
inline float position_gumbel(ulong seed,ulong pos,uint token,ulong purpose) {
  return -metal::log(-metal::log(position_uniform(seed,pos,token,purpose)));
}
#endif
